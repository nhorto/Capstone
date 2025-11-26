import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
class LSTMForecastModel:
    def __init__(self, data=None, file_path=None, lookback=5): # lookback is how many rows back in the data the model is seeing. you will need to adjust this if you
                                                              # adjsut how you resample the data (I commented on this below in the preprocess data method)
        """
        Initialize the LSTM model with either a DataFrame or a CSV file path.
        Parameters:
        - data: pandas DataFrame containing the input data.
        - file_path: path to the CSV file containing the data.
        - lookback: Number of past timesteps to use as input for LSTM.
        """
        if data is not None:
            self.df = data.copy()
        elif file_path is not None:
            self.df = pd.read_csv(file_path)
        else:
            raise ValueError("Either a DataFrame or a file path must be provided.")
        self.lookback = lookback
        self.scaler = MinMaxScaler()
        self.features = [
            'historical_power','average_temperature_C', 'WMS 01 irradiance', 'hours_since_sunrise','hours_until_sunset','peak_sun_factor','is_daytime',
            'total_last_active_fault', 'total_net_AC_voltage_V','total_time_run_today_h', 'average_average_cosphii_percent',
       'total_DC_voltage_DCV', 'total_output_current_A',
       'total_grid_connections','total_net_frequency_Hz'
        ]# is_daytime

        # Direct features to use (matching XGBoost)
        self.direct_features = ['WMS 01 irradiance','average_temperature_C', 'total_output_power_kW',
            'total_last_active_fault', 'total_net_AC_voltage_V','total_time_run_today_h', 'average_average_cosphii_percent',
       'total_DC_voltage_DCV', 'total_output_current_A',
       'total_grid_connections','total_net_frequency_Hz']

        # Define what features to engineer and how
        self.feature_engineering_plan = {
            'total_output_power_kW': {
                'lags': [1, 2, 3, 6, 12, 24],
                'rolling': {'mean': [1,3, 6, 12,24], 'std': [3,6], 'min': [3,6], 'max': [1,2,3,6]},
                'diff': [1, 2,3,4,5]
            },
            'average_temperature_C': {
                'lags': [1, 3, 6, 12],
                'rolling': {'mean': [1,2,3, 6,12,24], 'std': [2,3,6]},
                'diff': [1, 2,3]
            },
            'total_DC_voltage_DCV': {
                'lags': [1, 3, 6],
                'rolling': {'mean': [1,2,3, 6,12,24], 'std': [2,3,6]},
                'diff': [1, 2,3]
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

        if 'WMS 01 irradiance' in self.df.columns:
            self.feature_engineering_plan['WMS 01 irradiance'] = {
                'lags': [1, 2, 3, 6, 12, 24],
                'rolling': {'mean': [1,3, 6, 12], 'std': [2,3,6], 'max': [1,2, 3,6]},
                'diff': [1, 2, 3,4,5]
            }
        self.target = None

        '''
       for self.features you can add more columns to this if you want. but these are some of the ones that i have been using.
        '''

    def preprocess_data(self, resample_freq='30min', target_horizon=18):
        """
        Preprocess the dataset for LSTM input using XGBoost-style resampling.
        - resample_freq: frequency string (e.g., '10min', '30min')
        - target_horizon: number of rows (at original freq) to shift target forward
        """
        if 'datetime' in self.df.columns:
            self.df['datetime'] = pd.to_datetime(self.df['datetime'])
            self.df.set_index('datetime', inplace=True)
        elif 'index' in self.df.columns:
            self.df['datetime'] = pd.to_datetime(self.df['index'])
            self.df.set_index('datetime', inplace=True)

        self.df['hour'] = self.df.index.hour
        self.df['day'] = self.df.index.day
        self.df['month'] = self.df.index.month
        self.df['dayofweek'] = self.df.index.dayofweek
        self.df['quarter'] = self.df.index.quarter
        self.df['minute'] = self.df.index.minute

        self.df['hours_since_sunrise'] = (self.df['hour'] + self.df['minute'] / 60 - 6).clip(lower=0)
        self.df['hours_until_sunset'] = (18 - (self.df['hour'] + self.df['minute'] / 60)).clip(lower=0)
        self.df['is_daytime'] = ((self.df['hour'] >= 6) & (self.df['hour'] < 18)).astype(int)

        self.df['peak_sun_factor'] = np.sin(np.pi * (self.df['hour'] + self.df['minute']/60 - 6) / 12)
        self.df['peak_sun_factor'] = self.df['peak_sun_factor'].clip(lower=0)

        self.df['hour_sin'] = np.sin(2 * np.pi * self.df['hour'] / 24)
        self.df['hour_cos'] = np.cos(2 * np.pi * self.df['hour'] / 24)
        self.df['month_sin'] = np.sin(2 * np.pi * self.df['month'] / 12)
        self.df['month_cos'] = np.cos(2 * np.pi * self.df['month'] / 12)

        # Shift irradiance to simulate known-ahead forecast (matching XGBoost)
        self.df['irradiance_future'] = self.df['WMS 01 irradiance'].shift(-target_horizon)

        # Original target
        original_power = self.df['total_output_power_kW'].copy()

        # Resample non-target features
        df_features = self.df.drop(columns=['total_output_power_kW'])
        df_resampled = df_features.select_dtypes(include=[np.number]).resample(resample_freq).mean()

        # Resample power for input (to include historical context)
        df_resampled['historical_power'] = original_power.resample(resample_freq).mean()

        # Shift original target to create a future prediction target
        future_target = original_power.shift(-target_horizon)
        irradiance_future = self.df['irradiance_future']
        df_resampled['power_future'] = future_target.resample(resample_freq).mean()
        df_resampled['irradiance_future'] = irradiance_future.resample(resample_freq).mean()

        # Align future target and irradiance to df_resampled
        common_idx = df_resampled.index.intersection(future_target.index)
        df_resampled = df_resampled.loc[common_idx].copy()
        df_resampled['power_future'] = future_target.loc[common_idx]
        df_resampled['irradiance_future'] = df_resampled['irradiance_future'].loc[common_idx]

        # Feature engineering
        for col, ops in self.feature_engineering_plan.items():
            if col not in df_resampled.columns:
                print(f"Warning: {col} not found in dataframe, skipping")
                continue

            if 'lags' in ops:
                for lag in ops['lags']:
                    feature_name = f'{col}_lag_{lag}'
                    df_resampled[feature_name] = df_resampled[col].shift(lag)
                    self.features.append(feature_name)

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

            if 'diff' in ops:
                for periods in ops['diff']:
                    feature_name = f'{col}_diff_{periods}'
                    df_resampled[feature_name] = df_resampled[col].diff(periods=periods)
                    self.features.append(feature_name)

        # Create interaction features if solar irradiance is available (matching XGBoost)
        if 'WMS 01 irradiance' in df_resampled.columns:
            df_resampled['irr_temp_interaction'] = df_resampled['WMS 01 irradiance'] * df_resampled['average_temperature_C']
            df_resampled['irr_time_interaction'] = df_resampled['WMS 01 irradiance'] * df_resampled['peak_sun_factor']
            self.features += ['irr_temp_interaction', 'irr_time_interaction']

        # Add direct features if they exist in the dataframe (matching XGBoost)
        for feature in self.direct_features:
            if feature in df_resampled.columns and feature not in self.features:
                self.features.append(feature)

        self.df_clean = df_resampled.dropna()

        # Scale input features only
        self.df_clean[self.features] = self.scaler.fit_transform(self.df_clean[self.features])

        # add for later use in blender
        self.target = self.df_clean['power_future']

    def create_sequences(self):
        """Convert the dataframe into sequences for LSTM input."""
        X, y = [], []
        data = self.df_clean[self.features].values
        target = self.df_clean['power_future'].values
        for i in range(len(data) - self.lookback):
            X.append(data[i:i + self.lookback])
            y.append(target[i + self.lookback])
        self.X = np.array(X)
        self.y = np.array(y)

        # Train-test split (80% train, 20% test)
        split_idx = int(len(self.X) * 0.8)
        self.X_train, self.X_test = self.X[:split_idx], self.X[split_idx:]
        self.y_train, self.y_test = self.y[:split_idx], self.y[split_idx:]

    def build_model(self):
        """Define the LSTM model architecture."""
        model = Sequential([
            LSTM(128, activation='tanh', return_sequences=True, input_shape=(self.lookback, len(self.features))),
            Dropout(0.2),
            LSTM(64, activation='tanh'),
            Dropout(0.2),
            Dense(50, activation='relu'),
            Dense(25, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
        self.model = model

    def train_model(self, epochs=50, batch_size=16):
        """Train the LSTM model and store history for plotting."""
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

        self.history = self.model.fit(
            self.X_train, self.y_train, epochs=epochs, batch_size=batch_size,
            validation_data=(self.X_test, self.y_test), verbose=1, callbacks=[early_stopping]
        )

    def evaluate_model(self):
        """Evaluate the LSTM model on the test set."""
        y_pred = self.model.predict(self.X_test)
        y_pred = np.clip(y_pred, a_min=0, a_max=None)  # for non-negative predictions
        self.metrics = {
            'rmse': np.sqrt(mean_squared_error(self.y_test, y_pred)),
            'mae': mean_absolute_error(self.y_test, y_pred),
            'r2': r2_score(self.y_test, y_pred)
        }
        return self.metrics

    def plot_results(self):
        """Plot actual vs predicted power output."""
        y_pred = self.model.predict(self.X_test)
        y_pred = np.clip(y_pred, a_min=0, a_max=None)
        plt.figure(figsize=(12, 6))
        plt.plot(self.y_test, label='Actual', alpha=0.8)
        plt.plot(y_pred, label='Predicted', alpha=0.8)
        plt.title('Hourly Solar Power Prediction (LSTM, 3 hours ahead)')
        plt.ylabel('Power Output (kW)')
        plt.xlabel('Time')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

    def plot_training_history(self):
        """Plot training & validation loss and MAE."""
        fig, ax1 = plt.subplots(figsize=(12, 6))
        # Plot loss
        ax1.plot(self.history.history['loss'], label='Train Loss', color='blue')
        ax1.plot(self.history.history['val_loss'], label='Validation Loss', color='blue', linestyle='dashed')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss (MSE)')
        ax1.set_title('Training & Validation Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        plt.show()
        # Plot MAE
        fig, ax2 = plt.subplots(figsize=(12, 6))
        ax2.plot(self.history.history['mae'], label='Train MAE', color='red')
        ax2.plot(self.history.history['val_mae'], label='Validation MAE', color='red', linestyle='dashed')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('MAE')
        ax2.set_title('Training & Validation MAE')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        plt.show()

    def predict(self, new_data, resample_freq='30min', target_horizon=18):
        """
        Generate predictions on new, preprocessed data.

        Parameters:
        - new_data: pandas DataFrame with original structure (before resampling).
        - resample_freq: resampling frequency to match training.
        - target_horizon: number of future steps predicted ahead.

        Returns:
        - predictions: pandas Series with timestamps aligned.
        """
        if self.model is None:
            raise ValueError("Model not built or trained yet. Call train_model() first.")
        if not hasattr(self, 'features') or not self.features:
            raise ValueError("Feature list not found. Ensure preprocess_data() was called before training.")

        # Store original features to avoid modifying them
        original_features = self.features.copy()

        # Clone and preprocess new data
        temp_df = new_data.copy()

        if 'datetime' in temp_df.columns:
            temp_df['datetime'] = pd.to_datetime(temp_df['datetime'])
            temp_df.set_index('datetime', inplace=True)
        elif 'index' in temp_df.columns:
            temp_df['datetime'] = pd.to_datetime(temp_df['index'])
            temp_df.set_index('datetime', inplace=True)

        temp_df['hour'] = temp_df.index.hour
        temp_df['day'] = temp_df.index.day
        temp_df['month'] = temp_df.index.month
        temp_df['dayofweek'] = temp_df.index.dayofweek
        temp_df['quarter'] = temp_df.index.quarter
        temp_df['minute'] = temp_df.index.minute

        temp_df['hours_since_sunrise'] = (temp_df['hour'] + temp_df['minute'] / 60 - 6).clip(lower=0)
        temp_df['hours_until_sunset'] = (18 - (temp_df['hour'] + temp_df['minute'] / 60)).clip(lower=0)
        temp_df['is_daytime'] = ((temp_df['hour'] >= 6) & (temp_df['hour'] < 18)).astype(int)

        temp_df['peak_sun_factor'] = np.sin(np.pi * (temp_df['hour'] + temp_df['minute']/60 - 6) / 12)
        temp_df['peak_sun_factor'] = temp_df['peak_sun_factor'].clip(lower=0)

        temp_df['hour_sin'] = np.sin(2 * np.pi * temp_df['hour'] / 24)
        temp_df['hour_cos'] = np.cos(2 * np.pi * temp_df['hour'] / 24)
        temp_df['month_sin'] = np.sin(2 * np.pi * temp_df['month'] / 12)
        temp_df['month_cos'] = np.cos(2 * np.pi * temp_df['month'] / 12)

        # Shift irradiance to simulate known-ahead forecast (matching XGBoost)
        temp_df['irradiance_future'] = temp_df['WMS 01 irradiance'].shift(-target_horizon)

        original_power = temp_df['total_output_power_kW'].copy()

        df_features = temp_df.drop(columns=['total_output_power_kW'])
        df_resampled = df_features.select_dtypes(include=[np.number]).resample(resample_freq).mean()

        df_resampled['historical_power'] = original_power.resample(resample_freq).mean()

        future_target = original_power.shift(-target_horizon)
        irradiance_future = temp_df['irradiance_future']
        df_resampled['power_future'] = future_target.resample(resample_freq).first()
        df_resampled['irradiance_future'] = irradiance_future.resample(resample_freq).mean()

        common_idx = df_resampled.index.intersection(future_target.index)
        df_resampled = df_resampled.loc[common_idx].copy()
        df_resampled['power_future'] = future_target.loc[common_idx]
        df_resampled['irradiance_future'] = df_resampled['irradiance_future'].loc[common_idx]

        for col, ops in self.feature_engineering_plan.items():
            if col not in df_resampled.columns:
                print(f"Warning: {col} not found in dataframe, skipping")
                continue

            if 'lags' in ops:
                for lag in ops['lags']:
                    feature_name = f'{col}_lag_{lag}'
                    df_resampled[feature_name] = df_resampled[col].shift(lag)

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

            if 'diff' in ops:
                for periods in ops['diff']:
                    feature_name = f'{col}_diff_{periods}'
                    df_resampled[feature_name] = df_resampled[col].diff(periods=periods)

        # Create interaction features if solar irradiance is available (matching XGBoost)
        if 'WMS 01 irradiance' in df_resampled.columns:
            df_resampled['irr_temp_interaction'] = df_resampled['WMS 01 irradiance'] * df_resampled['average_temperature_C']
            df_resampled['irr_time_interaction'] = df_resampled['WMS 01 irradiance'] * df_resampled['peak_sun_factor']

        df_resampled.dropna(inplace=True)

        missing_features = [f for f in original_features if f not in df_resampled.columns]
        if missing_features:
            raise ValueError(f"Missing features in prediction data: {missing_features}")

        df_resampled[original_features] = self.scaler.transform(df_resampled[original_features])

        # Create sequences
        X_pred = []
        index = []
        data = df_resampled[original_features].values
        for i in range(len(data) - self.lookback - target_horizon + 1):
            X_pred.append(data[i:i + self.lookback])
            index.append(df_resampled.index[i + self.lookback + target_horizon - 1])  # Align prediction with future

        X_pred = np.array(X_pred)

        if len(X_pred) == 0:
            raise ValueError("Not enough data to generate sequences for prediction.")

        predictions = self.model.predict(X_pred)
        predictions = np.clip(predictions, a_min=0, a_max=None)

        return pd.Series(predictions.flatten(), index=index, name='lstm_predictions')


    def run_pipeline(self, epochs=100, batch_size=64):
        """Run the full pipeline: preprocessing, sequence creation, training, evaluating, and plotting."""
        self.preprocess_data()
        self.create_sequences()
        self.build_model()
        self.train_model(epochs=epochs, batch_size=batch_size)
        results = self.evaluate_model()
        self.plot_results()
        self.plot_training_history()
        return results