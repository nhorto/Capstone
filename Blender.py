from XGBoost import XGBoost
from LSTM import LSTMForecastModel
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class blender:
    def __init__(self, data, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
        self.data = data
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.split_data()

    def split_data(self):
        total_len = len(self.data)
        train_end = int(self.train_ratio * total_len)
        val_end = train_end + int(self.val_ratio * total_len)

        # Use .iloc for positional slicing but reset_index to preserve datetime info
        self.base_train_data = self.data.iloc[:train_end].reset_index()
        self.holdout_data = self.data.iloc[train_end:val_end].reset_index()
        self.test_data = self.data.iloc[val_end:].reset_index()
        
        # If the original data had a datetime index, rename it appropriately
        if isinstance(self.data.index, pd.DatetimeIndex):
            self.base_train_data = self.base_train_data.rename(columns={'index': 'datetime'})
            self.holdout_data = self.holdout_data.rename(columns={'index': 'datetime'})
            self.test_data = self.test_data.rename(columns={'index': 'datetime'})

    def run_models(self, lstm_freq='60min', xgb_freq='1H', lstm_horizon=18, xgb_horizon=18):
        # === Train LSTM ===
        self.lstm_model = LSTMForecastModel(data=self.base_train_data)
        self.lstm_model.preprocess_data(resample_freq=lstm_freq, target_horizon=lstm_horizon)
        self.lstm_model.create_sequences()
        self.lstm_model.build_model()
        self.lstm_model.train_model()
        self.lstm_model.evaluate_model()

        # === LSTM Predictions ===
        self.lstm_holdout = self.lstm_model.predict(
            new_data=self.holdout_data,
            resample_freq=lstm_freq,
            target_horizon=lstm_horizon
        )

        self.y_lstm_test = self.lstm_model.predict(
            new_data=self.test_data,
            resample_freq=lstm_freq,
            target_horizon=lstm_horizon
        )

        # === Train XGBoost ===
        self.xgb_model = XGBoost(data=self.base_train_data.reset_index())
        self.xgb_model.preprocess_data(resample_freq=xgb_freq, target_horizon=xgb_horizon)
        self.xgb_model.train_model()
        self.xgb_model.evaluate_model()

        # === XGBoost Predictions ===
        # Create new model for holdout data
        xgb_holdout_model = XGBoost(data=self.holdout_data.reset_index())
        xgb_holdout_model.preprocess_data(resample_freq=xgb_freq, target_horizon=xgb_horizon)
        #reuse the trained model and scaler
        xgb_holdout_model.model = self.xgb_model.model
        xgb_holdout_model.scaler = self.xgb_model.scaler
        # predictions
        self.xgb_holdout = xgb_holdout_model.predict(data=xgb_holdout_model.df_clean)

        # Create new model for the test data
        xgb_test_model = XGBoost(data=self.test_data.reset_index())
        xgb_test_model.preprocess_data(resample_freq=xgb_freq, target_horizon=xgb_horizon)
        xgb_test_model.model = self.xgb_model.model
        xgb_test_model.scaler = self.xgb_model.scaler
        # predict
        self.xgb_test = xgb_test_model.predict(data=xgb_test_model.df_clean)

        # === Align predictions and true values on holdout ===
        y_true_holdout = xgb_holdout_model.target
        common_index = y_true_holdout.index.intersection(self.xgb_holdout.index).intersection(self.lstm_holdout.index)

        self.blend_df = pd.DataFrame({
            'y_true': y_true_holdout.loc[common_index],
            'lstm_pred': self.lstm_holdout.loc[common_index].values,
            'xgb_pred': self.xgb_holdout.loc[common_index].values
        }, index=common_index)

        # Align test set predictions for blended model
        y_true_test = xgb_test_model.target
        common_test_index = y_true_test.index.intersection(self.y_lstm_test.index).intersection(self.xgb_test.index)

        X_blend_df = pd.DataFrame({
            'lstm_pred': self.y_lstm_test.loc[common_test_index].values,
            'xgb_pred': self.xgb_test.loc[common_test_index].values
        }, index=common_test_index)

        y_true_aligned = y_true_test.loc[common_test_index]

        self.X_blend_test = X_blend_df
        self.y_blend_true = y_true_aligned

    def train_blended_model(self):
        from sklearn.linear_model import Ridge
        from sklearn.linear_model import RidgeCV
        from sklearn.preprocessing import StandardScaler

        X_blend = self.blend_df[['lstm_pred', 'xgb_pred']]
        y_blend = self.blend_df['y_true']

        # scaler = StandardScaler()
        # X_blend_scaled = scaler.fit_transform(X_blend)

        self.blend_model = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0])
        self.blend_model.fit(X_blend, y_blend)

        # Align final test set inputs
        X_blend_df = self.X_blend_test[['lstm_pred', 'xgb_pred']]
        # X_blend_df_scaled = scaler.transform(X_blend_df)

        self.blend_model_pred = pd.Series(
            self.blend_model.predict(X_blend_df),
            index=X_blend_df.index
        )

        # self.blend_model_pred = self.blend_model_pred[self.X_blend_test.index.get_indexer(features.index)]

        return self.blend_model_pred

    def _generate_time_features(self, index):
        """
        Generates time-based features given a datetime index.

        Parameters:
        - index: pd.DatetimeIndex used to generate features.

        Returns:
        - DataFrame of time features.
        """
        features = pd.DataFrame(index=index)
        features['hour'] = index.hour
        features['minute'] = index.minute
        features['dayofweek'] = index.dayofweek  # 0=Monday, 6=Sunday
        features['is_weekend'] = index.dayofweek >= 5
        features['month'] = index.month
        features['dayofyear'] = index.dayofyear
        features['weekofyear'] = index.isocalendar().week.astype(int)
        features['quarter'] = index.quarter

        return features

    def _generate_residual_features(self, max_lag=5):
        """
        Generate features for residual correction including time-based and lag features.

        Parameters:
        - max_lag: Number of lags to generate for predictions and residuals.

        Returns:
        - DataFrame of features aligned with the holdout set (blend_df).
        """
        df = self.blend_df.copy()

        # Step 1: Time features
        time_features = self._generate_time_features(df.index)

        # Step 2: Lag features for predictions and residuals
        for lag in range(1, max_lag + 1):
            time_features[f'lstm_pred_lag_{lag}'] = df['lstm_pred'].shift(lag)
            time_features[f'xgb_pred_lag_{lag}'] = df['xgb_pred'].shift(lag)

            # Optionally, lag the residuals themselves
            # residuals = df['y_true'] - df[['lstm_pred', 'xgb_pred']].mean(axis=1)
            # time_features[f'residual_lag_{lag}'] = residuals.shift(lag)


        # Drop rows with NaNs from shifting
        time_features.dropna(inplace=True)

        # Also drop rows in blend_df to keep alignment
        self.blend_df = self.blend_df.loc[time_features.index]

        return time_features

    def train_residual_model(self):
        """
        Trains a model to predict the residual error from the meta-model,
        using time-based features.
        """
        from xgboost import XGBRegressor

        # compute residuals
        residual_target = self.blend_df['y_true'] - self.blend_model.predict(self.blend_df[['lstm_pred', 'xgb_pred']])

        # create time-based features with blend_df index
        time_features = self._generate_time_features(self.blend_df.index)
        residual_features = self._generate_residual_features(max_lag=3)

        print(self.holdout_data.columns)
        original_features = self.holdout_data['WMS 01 irradiance']
        original_features = original_features.loc[self.blend_df.index]  # align
        residual_features = pd.concat([residual_features, original_features], axis=1)

        # train residual model
        self.residual_model = XGBRegressor()
        # self.residual_model.fit(time_features, residual_target)
        self.residual_model.fit(residual_features, residual_target.loc[residual_features.index])

        # Save time_features for test correction
        self.residual_train_features = time_features

    def apply_residual_correction(self, max_lag=3):
        """
        Applies residual correction on top of the meta-model predictions,
        using time-based features from the test set.
        """

        df_test = self.X_blend_test.copy()

        # Step 1: generate time-based features
        features = self._generate_time_features(df_test.index)

        # Step 2: add lag features
        for lag in range(1, max_lag + 1):
            features[f'lstm_pred_lag_{lag}'] = df_test['lstm_pred'].shift(lag)
            features[f'xgb_pred_lag_{lag}'] = df_test['xgb_pred'].shift(lag)

        # Step 3: add external features
        extra_features = self.test_data['WMS 01 irradiance']
        extra_features = extra_features.loc[features.index]
        features = pd.concat([features, extra_features], axis=1)

        # Step 4: Drop NaNs from shifting
        features.dropna(inplace=True)
        self.X_blend_test = self.X_blend_test.loc[features.index]
        self.y_blend_true = self.y_blend_true.loc[features.index]
        self.blend_model_pred = self.blend_model_pred[features.index]

        # Step 5: predict residuals & correct
        residual_pred = self.residual_model.predict(features)
        corrected_preds = self.blend_model_pred + residual_pred

        # Step 6: clip negative values
        corrected_preds = np.clip(corrected_preds, 0, None)

        # Step 7: optionally force nighttime values to 0
        # night_mask = self.test_data.loc[corrected_preds.index, 'is_daytime'] == 0
        # corrected_preds[night_mask] = 0

        # Update prediction
        self.blend_model_pred = corrected_preds

        return corrected_preds


    def evaluate_blend_model(self, y_test_true):
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

        # align actual with prediction index
        y_true = y_test_true.loc[self.X_blend_test.index]
        y_pred = self.blend_model_pred

        # metrics
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)

        print(f"Meta-Model Evaluation:\nMAE: {mae:.3f} | RMSE: {rmse:.3f} | R²: {r2:.3f}")

        # Plot actual vs predicted
        plt.figure(figsize=(12, 6))
        plt.plot(y_true.index, y_true.values, label='Actual', linewidth=2)
        plt.plot(y_true.index, y_pred, label='Meta Predicted', linestyle='--')
        plt.title("Meta Model: Actual vs Predicted")
        plt.xlabel("Time")
        plt.ylabel("Power Output (kW)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        return mae, rmse, r2