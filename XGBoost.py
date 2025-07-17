import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV,RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
import seaborn as sns

class XGBoostModel:
    def __init__(self, data=None, file_path=None, direct_features=None):
        if data is not None:
            self.df = data.copy()
        elif file_path is not None:
            self.df = pd.read_csv(file_path)
        else:
            raise ValueError("Provide either a DataFrame or file path")

        # Allow user to specify which features to use directly
        #self.direct_features = direct_features or ['WMS 01 irradiancee']

        self.scaler = MinMaxScaler()
        self.model = None

        self.target = None

        # Starting with basic features
        self.features = []

        # Direct features to use (these are passed directly to the model without transformations)
        self.direct_features = ['WMS 01 irradiance','average_temperature_C', 'total_output_power_kW',
            'total_last_active_fault', 'total_net_AC_voltage_V','total_time_run_today_h', 'average_average_cosphii_percent',
       'total_DC_voltage_DCV', 'total_output_current_A',
       'total_grid_connections','total_net_frequency_Hz']

        # Define what features to engineer and how
        self.feature_engineering_plan = {
            'total_output_power_kW': {
                'lags': [1, 2, 3, 6, 12, 24],  # Expanded lag windows
                'rolling': {'mean': [1,3, 6, 12,24], 'std': [3,6], 'min': [3,6], 'max': [1,2,3,6]},
                'diff': [1, 2,3,4,5]  # Rate of change in power
            },
            'average_temperature_C': {
                'lags': [1, 3, 6, 12],
                'rolling': {'mean': [1,2,3, 6,12,24], 'std': [2,3,6]},
                'diff': [1, 2,3]  # Temperature change rate
            },
            'total_DC_voltage_DCV': {
                'lags': [1, 3, 6],
                'rolling': {'mean': [1,2,3, 6,12,24], 'std': [2,3,6]},
                'diff': [1, 2,3]  # Voltage change rate
            },
            'total_output_current_A': {
                'lags': [1, 3, 6],
                'rolling': {'mean': [1,2,3, 6,12,24], 'std': [2,3,6]},
                'diff': [1, 2,3]
            },
            'average_average_cosphii_percent': {
                'lags': [1, 3, 6],
                'rolling': {'mean': [1,2,3, 6,12,24]},
                'diff': [1, 2,3]
            },
            'total_time_run_today_h': {
                'lags': [1, 3, 6],
                'rolling': {'sum': [1,3, 6,12,24]},
                'diff': [1, 2,3]
            },
            'total_last_active_fault': {
                'lags': [1, 3],
                'rolling': {'sum': [1,3, 6,12,24]},
                'diff': [1, 2,3]
            }
        }

        # If solar irradiance is available, add it to plan
        if 'WMS 01 irradiance' in self.df.columns:
            self.feature_engineering_plan['WMS 01 irradiance'] = {
                'lags': [0, 1, 2, 3, 6, 12, 24],
                'rolling': {'mean': [1,3, 6, 12], 'std': [2,3,6], 'max': [1,2, 3,6]},
                'diff': [1, 2, 3,4,5]  # Rate of change in irradiance
            }

    def preprocess_data(self, target_horizon=18, resample_freq='10T'):
        """
        Preprocess data with more sophisticated feature engineering

        Parameters:
        - target_horizon: Number of periods ahead to predict
        - resample_freq: Frequency to resample data ('60T' for hourly)
        """
        # Parse datetime
        if 'datetime' in self.df.columns:
            self.df['datetime'] = pd.to_datetime(self.df['datetime'])
            self.df.set_index('datetime', inplace=True)
        elif 'index' in self.df.columns:
            self.df['datetime'] = pd.to_datetime(self.df['index'])
            self.df.set_index('datetime', inplace=True)
        else:
          # If no datetime column is found, try to infer from 'date' and 'time'
          try:
              self.df['datetime'] = pd.to_datetime(self.df['date'] + ' ' + self.df['time'])
              self.df.set_index('datetime', inplace=True)
          except KeyError:
              raise ValueError("No valid datetime information found. Please provide a 'datetime', 'date' and 'time', or 'index' column.")

        # Enhanced time-based features
        self.df['hour'] = self.df.index.hour
        self.df['day'] = self.df.index.day
        self.df['month'] = self.df.index.month
        self.df['dayofweek'] = self.df.index.dayofweek
        self.df['quarter'] = self.df.index.quarter
        self.df['minute'] = self.df.index.minute

        # Solar position-related features (improved)
        self.df['hours_since_sunrise'] = (self.df['hour'] + self.df['minute'] / 60 - 6).clip(lower=0)
        self.df['hours_until_sunset'] = (18 - (self.df['hour'] + self.df['minute'] / 60)).clip(lower=0)
        self.df['is_daytime'] = ((self.df['hour'] >= 6) & (self.df['hour'] < 18)).astype(int)

        # Peak sun hours approximation (simplified model)
        self.df['peak_sun_factor'] = np.sin(np.pi * (self.df['hour'] + self.df['minute']/60 - 6) / 12)
        self.df['peak_sun_factor'] = self.df['peak_sun_factor'].clip(lower=0)

        # Create cyclic time features to better capture periodicity
        self.df['hour_sin'] = np.sin(2 * np.pi * self.df['hour'] / 24)
        self.df['hour_cos'] = np.cos(2 * np.pi * self.df['hour'] / 24)
        self.df['month_sin'] = np.sin(2 * np.pi * self.df['month'] / 12)
        self.df['month_cos'] = np.cos(2 * np.pi * self.df['month'] / 12)

        # Step 0: Shift irradiance to simulate known-ahead forecast
        self.df['irradiance_future'] = self.df['WMS 01 irradiance'].shift(-target_horizon)

        # Step 1: Keep the original high-frequency target series
        original_target = self.df['total_output_power_kW'].copy()
        future_target = original_target.shift(-target_horizon)
        irradiance_future = self.df['irradiance_future']

        # Step 2: Resample all numeric feature columns except the target
        df_features = self.df.drop(columns=['total_output_power_kW'])
        df_resampled = df_features.select_dtypes(include=[np.number]).resample(resample_freq).mean()

        # Step 3: Resample future target and future irradiance to match resample frequency
        df_resampled['power_future'] = future_target.resample(resample_freq).first()
        df_resampled['irradiance_future'] = irradiance_future.resample(resample_freq).first()

        # Step 4: Align everything using a common index
        common_idx = df_resampled.index.intersection(df_resampled['power_future'].dropna().index)
        df_resampled = df_resampled.loc[common_idx].copy()
        df_resampled['power_future'] = df_resampled['power_future'].loc[common_idx]
        df_resampled['irradiance_future'] = df_resampled['irradiance_future'].loc[common_idx]

        # Feature engineering based on plan
        for col, ops in self.feature_engineering_plan.items():
            if col not in df_resampled.columns:
                print(f"Warning: {col} not found in dataframe, skipping")
                continue

            # Create lag features
            if 'lags' in ops:
                for lag in ops['lags']:
                    feature_name = f'{col}_lag_{lag}'
                    df_resampled[feature_name] = df_resampled[col].shift(lag)
                    self.features.append(feature_name)

            # Create rolling window features
            if 'rolling' in ops:
                for method, windows in ops['rolling'].items():
                    for window in windows:
                        feature_name = f'{col}_{method}_{window}'
                        if method == 'mean':
                            df_resampled[feature_name] = df_resampled[col].rolling(window=window).mean()
                        elif method == 'sum':
                            df_resampled[feature_name] = df_resampled[col].rolling(window=window).sum()
                        elif method == 'std':
                            df_resampled[feature_name] = df_resampled[col].rolling(window=window).std()
                        elif method == 'min':
                            df_resampled[feature_name] = df_resampled[col].rolling(window=window).min()
                        elif method == 'max':
                            df_resampled[feature_name] = df_resampled[col].rolling(window=window).max()
                        self.features.append(feature_name)

            # Create difference features (rate of change)
            if 'diff' in ops:
                for periods in ops['diff']:
                    feature_name = f'{col}_diff_{periods}'
                    df_resampled[feature_name] = df_resampled[col].diff(periods=periods)
                    self.features.append(feature_name)

        # Add time features to feature list
        time_features = ['hour', 'hour_sin', 'hour_cos', 'month_sin', 'month_cos',
                         'hours_since_sunrise', 'hours_until_sunset', 'is_daytime',
                         'peak_sun_factor']

        df_resampled[time_features] = self.df[time_features]
        self.features += time_features

        # Create interaction features if solar irradiance is available
        if 'WMS 01 irradiance' in df_resampled.columns:
            df_resampled['irr_temp_interaction'] = df_resampled['WMS 01 irradiance'] * df_resampled['average_temperature_C']
            df_resampled['irr_time_interaction'] = df_resampled['WMS 01 irradiance'] * df_resampled['peak_sun_factor']
            self.features += ['irr_temp_interaction', 'irr_time_interaction']

        # Create target (future power output)
        #df_resampled['power_future'] = df_resampled['total_output_power_kW'].shift(-target_horizon)

        # Drop rows with NaN values
        self.df_clean = df_resampled.dropna()

        # Add for later use in metaModel
        self.target = self.df_clean['power_future']

        # Add direct features if they exist in the dataframe
        for feature in self.direct_features:
            if feature in df_resampled.columns:
                self.features.append(feature)
                print(f"Added direct feature: {feature}")

        # Print final feature list count
        print(f"Total features created: {len(self.features)}")
        print(f"THES ARE THE COLUMNS: {[col for col in self.df_clean.columns]}")

    def analyze_features(self):
        """Analyze feature importance and relationships"""
        X = self.df_clean[self.features]
        y = self.df_clean['power_future']

        # Correlation heatmap of top features with target
        corr = pd.DataFrame(X.corrwith(y).sort_values(ascending=False)).reset_index()
        corr.columns = ['Feature', 'Correlation']

        plt.figure(figsize=(10, 12))
        top_n = min(20, len(corr))
        sns.barplot(x='Correlation', y='Feature', data=corr.head(top_n))
        plt.title(f'Top {top_n} Feature Correlations with Target')
        plt.tight_layout()
        plt.show()

        # Plot relationship between top features and target
        top_features = corr.head(5)['Feature'].tolist()

        fig, axes = plt.subplots(len(top_features), 1, figsize=(12, 15))
        for i, feature in enumerate(top_features):
            axes[i].scatter(X[feature], y, alpha=0.4)
            axes[i].set_xlabel(feature)
            axes[i].set_ylabel('Future Power Output')
            axes[i].set_title(f'{feature} vs Target')
            axes[i].grid(True)

        plt.tight_layout()
        plt.show()

        return corr

    def get_important_features(self, top_n=None, threshold=None):
        """
        Get a list of important features based on trained model.

        Parameters:
        - top_n: Select the top N features by importance.
        - threshold: Select features with importance above this threshold (e.g., 0.01).

        Returns:
        - List of selected feature names
        """
        if self.model is None:
            raise ValueError("Model must be trained first to compute feature importance.")

        importance_df = pd.DataFrame({
            'Feature': self.features,
            'Importance': self.model.feature_importances_
        })

        if top_n is not None:
            selected = importance_df.sort_values('Importance', ascending=False).head(top_n)['Feature'].tolist()
        elif threshold is not None:
            selected = importance_df[importance_df['Importance'] >= threshold]['Feature'].tolist()
        else:
            raise ValueError("Provide either top_n or threshold.")

        return selected

    def train_model(self, selected_features=None, param_grid=None):
        """Train XGBoost model with improved grid search"""
        if selected_features:
            used_features = [f for f in selected_features if f in self.features]
        else:
            used_features = self.features

        X = self.df_clean[used_features]
        y = self.df_clean['power_future']

        # Train-test split with stratified temporal split
        split_idx = int(len(self.df_clean) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        self.X_train, self.X_test = X_train, X_test
        self.y_train, self.y_test = y_train, y_test

        # Scale features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)

        # # Default param grid if none provided
        # if param_grid is None:
        #     param_grid = {
        #         'n_estimators': [100, 200, 300],
        #         'max_depth': [5, 7, 9],
        #         'learning_rate': [0.01, 0.05, 0.1],
        #         'subsample': [0.8, 0.9, 1.0],
        #         'colsample_bytree': [0.8, 0.9, 1.0],
        #         'min_child_weight': [1, 3, 5],
        #         'gamma': [0, 0.1, 0.2]
        #     }

        # # Grid search with cross-validation
        # grid = RandomizedSearchCV(
        #     XGBRegressor(objective='reg:squarederror', random_state=42),
        #     param_distributions=param_grid,
        #     n_iter=30,  # Try 20 random combinations
        #     cv=3,
        #     scoring='neg_mean_squared_error',
        #     verbose=1,
        #     n_jobs=-1
        # )

        # grid.fit(self.X_train_scaled, self.y_train)
        # self.model = grid.best_estimator_

        # print(f"Best parameters: {grid.best_params_}")

        default_params = {
            'n_estimators': 300,
            'max_depth': 5,
            'learning_rate': 0.01,
            'subsample': 0.9,
            'colsample_bytree': 0.8,
            'min_child_weight': 1,
            'gamma': 0,
            'objective': 'reg:quantileerror',
            'quantile_alpha': 0.80,
            'random_state': 42
        }

        # Initialize and fit the model
        self.model = XGBRegressor(**default_params)
        self.model.fit(self.X_train_scaled, self.y_train)

        # Get feature importance
        feature_importance = pd.DataFrame({
            'Feature': used_features,
            'Importance': self.model.feature_importances_
        }).sort_values('Importance', ascending=False)

        plt.figure(figsize=(10, 12))
        sns.barplot(x='Importance', y='Feature', data=feature_importance.head(20))
        plt.title('Feature Importance (Top 20)')
        plt.tight_layout()
        plt.show()

        return feature_importance

    def evaluate_model(self):
        """Evaluate model with additional metrics"""
        # Make predictions
        y_pred_train = self.model.predict(self.X_train_scaled)
        y_pred_test = self.model.predict(self.X_test_scaled)

        # Clip negative predictions to zero
        y_pred_train = np.clip(y_pred_train, a_min=0, a_max=None)
        y_pred_test = np.clip(y_pred_test, a_min=0, a_max=None)

        # Calculate metrics
        train_metrics = {
            'rmse': np.sqrt(mean_squared_error(self.y_train, y_pred_train)),
            'mae': mean_absolute_error(self.y_train, y_pred_train),
            'r2': r2_score(self.y_train, y_pred_train)
        }

        test_metrics = {
            'rmse': np.sqrt(mean_squared_error(self.y_test, y_pred_test)),
            'mae': mean_absolute_error(self.y_test, y_pred_test),
            'r2': r2_score(self.y_test, y_pred_test)
        }

        # Calculate normalized metrics
        y_mean = self.y_test.mean()
        test_metrics['nrmse'] = test_metrics['rmse'] / y_mean
        test_metrics['nmae'] = test_metrics['mae'] / y_mean

        print("\nTraining Metrics:")
        for k, v in train_metrics.items():
            print(f"{k.upper()}: {v:.4f}")

        print("\nTest Metrics:")
        for k, v in test_metrics.items():
            print(f"{k.upper()}: {v:.4f}")

        self.metrics = test_metrics
        return train_metrics, test_metrics

    def plot_results(self):
        """Plot comprehensive evaluation results"""
        y_pred = self.model.predict(self.X_test_scaled)
        y_pred = np.clip(y_pred, a_min=0, a_max=None)

        # Create a DataFrame with actual and predicted values
        results_df = pd.DataFrame({
            'Actual': self.y_test.values,
            'Predicted': y_pred
        }, index=self.y_test.index)

        # Plot actual vs predicted time series
        plt.figure(figsize=(15, 6))
        plt.plot(results_df.index, results_df['Actual'], label='Actual', color='blue', alpha=0.7)
        plt.plot(results_df.index, results_df['Predicted'], label='Predicted', color='red', alpha=0.7)
        plt.title('Solar Power Prediction - Actual vs Predicted')
        plt.ylabel('Power Output (kW)')
        plt.xlabel('Time')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # Plot scatter of actual vs predicted
        plt.figure(figsize=(10, 6))
        plt.scatter(results_df['Actual'], results_df['Predicted'], alpha=0.5)

        # Add perfect prediction line
        max_val = max(results_df['Actual'].max(), results_df['Predicted'].max())
        plt.plot([0, max_val], [0, max_val], 'r--')

        plt.title('Actual vs Predicted Power Output')
        plt.xlabel('Actual Power (kW)')
        plt.ylabel('Predicted Power (kW)')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # Plot residuals
        residuals = results_df['Actual'] - results_df['Predicted']
        plt.figure(figsize=(15, 6))
        plt.subplot(1, 2, 1)
        plt.scatter(results_df['Predicted'], residuals, alpha=0.5)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.title('Residuals vs Predicted')
        plt.xlabel('Predicted Power (kW)')
        plt.ylabel('Residual (Actual - Predicted)')
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.hist(residuals, bins=30, alpha=0.7)
        plt.title('Residual Distribution')
        plt.xlabel('Residual Value')
        plt.ylabel('Frequency')
        plt.grid(True)

        plt.tight_layout()
        plt.show()

        # Plot predictions for a sample period (2 weeks)
        if len(results_df) > 14*24:  # If we have at least 2 weeks of data
            sample_period = results_df.iloc[-14*24:]  # Last 2 weeks

            plt.figure(figsize=(15, 6))
            plt.plot(sample_period.index, sample_period['Actual'], label='Actual', color='blue')
            plt.plot(sample_period.index, sample_period['Predicted'], label='Predicted', color='red')
            plt.title('2-Week Sample Period - Actual vs Predicted')
            plt.ylabel('Power Output (kW)')
            plt.xlabel('Time')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()

        # Display error over time of day
        results_df['hour'] = results_df.index.hour
        results_df['abs_error'] = abs(results_df['Actual'] - results_df['Predicted'])

        hour_error = results_df.groupby('hour')['abs_error'].mean()

        plt.figure(figsize=(12, 6))
        hour_error.plot(kind='bar')
        plt.title('Mean Absolute Error by Hour of Day')
        plt.xlabel('Hour of Day')
        plt.ylabel('Mean Absolute Error')
        plt.grid(True, axis='y')
        plt.tight_layout()
        plt.show()

    def predict(self, data=None):
        """
        Predict on the full dataset (no train-test split).
        Useful when using this model as a feature generator for meta-models.

        Parameters:
        - data: optional, custom DataFrame to predict on.
                If None, will use self.df_clean from previous preprocessing.

        Returns:
        - predictions: np.array of predicted values
        - index: index of predictions (datetime or row index)
        """
        # if self.model is None:
        #     raise ValueError("Model has not been trained yet. Call train_model() first.")

        # if data is None:
        #     if not hasattr(self, 'df_clean'):
        #         raise ValueError("No preprocessed data available. Call preprocess_data() first.")
        #     X_full = self.df_clean[self.features]
        #     index = self.df_clean.index
        # else:
        #     # Use passed data (must already have matching features)
        #     missing = [feat for feat in self.features if feat not in data.columns]
        #     if missing:
        #         raise ValueError(f"Missing required features: {missing}")
        if data is None:
            data = self.df_clean

        X_full = data[self.features]
        index = data.index

        X_scaled = self.scaler.transform(X_full)
        predictions = self.model.predict(X_scaled)
        predictions = np.clip(predictions, a_min=0, a_max=None)

        return pd.Series(predictions, index=index, name="xgb_predictions")


    def test_for_leakage(self):
        """Leakage test: Train on shuffled targets and check performance"""

        # Shuffle the target values (break any true pattern)
        y_train_shuffled = np.random.permutation(self.y_train)

        # Train a new XGBoost model on the same features but with shuffled targets
        shuffled_model = XGBRegressor(objective='reg:squarederror', random_state=42)
        shuffled_model.fit(self.X_train_scaled, y_train_shuffled)

        # Predict on the test set
        y_pred_test = shuffled_model.predict(self.X_test_scaled)

        # Evaluate performance on real test targets
        rmse = np.sqrt(mean_squared_error(self.y_test, y_pred_test))
        mae = mean_absolute_error(self.y_test, y_pred_test)
        r2 = r2_score(self.y_test, y_pred_test)

        print("\n=== Leakage Detection Test ===")
        print("Model trained on SHUFFLED target values:")
        print(f"Test RMSE: {rmse:.4f}")
        print(f"Test MAE: {mae:.4f}")
        print(f"Test R²: {r2:.4f}")

        if r2 > 0.2:
            print("⚠️ WARNING: Model is performing too well on shuffled data. Possible leakage!")
        else:
            print("✅ No obvious leakage detected.")

        return {'rmse': rmse, 'mae': mae, 'r2': r2}


    def run_pipeline(self, target_horizon=18, resample_freq='30T'):
        """Run the complete pipeline"""
        print("Step 1: Preprocessing data...")
        self.preprocess_data(target_horizon=target_horizon, resample_freq=resample_freq)

        print("\nStep 2: Analyzing features...")
        feature_corr = self.analyze_features()

        print("\nstep 3: Get top features...")
        model.train_model()
        top_features = model.get_important_features(top_n=30)

        print("\nTop 30 Features Used: ")
        for i, feat in enumerate(top_features, start=1):
            print(f"{i}. {feat}")

        print("\nStep 3: Training model...")
        feature_importance = self.train_model(selected_features = top_features)

        print("\nStep 4: Evaluating model...")
        train_metrics, test_metrics = self.evaluate_model()

        print("\nStep 5: Plotting results...")
        self.plot_results()

        return {
            'feature_correlation': feature_corr,
            'feature_importance': feature_importance,
            'metrics': test_metrics
        }

# Usage example
# if __name__ == "__main__":
    # Example of how to use the model with solar irradiance as a direct feature

    # Option 1: Using default direct features (solar_irradiance)
    # model = ImprovedXGBoostModel(data=df.reset_index())
    # results = model.run_pipeline()
    # model.train_model()
    # model.test_for_leakage()
    # model.test_for_leakage()
    # # Option 2: Specify custom direct features
    # model = ImprovedXGBoostModel(
    #     file_path="your_solar_data.csv",
    #     direct_features=['solar_irradiance']
    # )
    # results = model.run_pipeline()
