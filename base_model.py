import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd

class baseModel:
    def __init__(self, data=None, file_path=None):
        """

        """
        if data is not None:
            self.df = data.copy()
        elif file_path is not None:
            self.df = pd.read_csv(file_path)
        else:
            raise ValueError("Either a DataFrame or a file path must be provided.")

        self.model = LinearRegression()
        self.scaler = MinMaxScaler()
        self.features = [
            'power_lag_1', 'power_lag_3', 'power_lag_6', 
            'temp_lag_1', 'temp_lag_3', 'temp_lag_6', 
            'power_ma_3', 'power_ma_6', 'temp_ma_3', 
            'WMS 01 irradiance', 'hour', 'hours_since_sunrise', 'hours_until_sunset', 'is_daytime'
        ]
    
    def preprocess_data(self):
        """Preprocess the dataset."""
        if 'datetime' in self.df.columns:
            self.df['datetime'] = pd.to_datetime(self.df['datetime'])
            self.df.set_index('datetime', inplace=True)
        elif 'index' in self.df.columns:
            self.df['datetime'] = pd.to_datetime(self.df['index'])
            self.df.set_index('datetime', inplace=True)

        self.df['hour'] = self.df.index.hour
        self.df['minute'] = self.df.index.minute

        self.df['hours_since_sunrise'] = self.df['hour'] + self.df['minute'] / 60 - 6
        self.df['hours_since_sunrise'] = self.df['hours_since_sunrise'].apply(lambda x: max(0, x))

        self.df['hours_until_sunset'] = 18 - (self.df['hour'] + self.df['minute'] / 60)
        self.df['hours_until_sunset'] = self.df['hours_until_sunset'].apply(lambda x: max(0, x))

        self.df['is_daytime'] = ((self.df['hour'] >= 6) & (self.df['hour'] < 18)).astype(int)


        original_power = self.df['total_output_power_kW'].copy()

        df_features = self.df.drop(columns=['total_output_power_kW'])
        df_hourly = df_features.select_dtypes(include=[np.number]).resample('60T').mean()

        df_hourly['historical_power'] = original_power.resample('60T').mean()

        future_target = original_power.shift(-18)
        df_hourly['power_future'] = future_target.resample('60T').first()

        common_idx = df_hourly.index.intersection(future_target.index)
        df_hourly = df_hourly.loc[common_idx].copy()
        df_hourly['power_future'] = future_target.loc[common_idx]

        # Rolling features
        df_hourly['power_ma_3'] = df_hourly['historical_power'].rolling(window=3).mean()
        df_hourly['power_ma_6'] = df_hourly['historical_power'].rolling(window=6).mean()
        df_hourly['temp_ma_3'] = df_hourly['average_temperature_C'].rolling(window=3).mean()

        # Lag features
        for lag in [1, 3, 6]:
            df_hourly[f'power_lag_{lag}'] = df_hourly['historical_power'].shift(lag)
            df_hourly[f'temp_lag_{lag}'] = df_hourly['average_temperature_C'].shift(lag)

        # Drop NaN values
        self.df_clean = df_hourly.dropna()

    def train_model(self):
        """Train the linear regression model."""
        X = self.df_clean[self.features]
        y = self.df_clean['power_future']

        # Train-test split (80% train, 20% test)
        split_idx = int(len(X) * 0.8)
        self.X_train, self.X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        self.y_train, self.y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        # Scale features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)

        # Train the model
        self.model.fit(self.X_train_scaled, self.y_train)

    def evaluate_model(self):
        """Evaluate the model and return performance metrics."""
        y_pred = self.model.predict(self.X_test_scaled)
        y_pred = np.clip(y_pred, a_min=0, a_max=None)  # clip to zero to get non-negative predictions

        self.metrics = {
            'rmse': np.sqrt(mean_squared_error(self.y_test, y_pred)),
            'mae': mean_absolute_error(self.y_test, y_pred),
            'r2': r2_score(self.y_test, y_pred)
        }

        # Feature importance
        importance = pd.DataFrame({
            'Feature': self.features,
            'Importance': np.abs(self.model.coef_)
        }).sort_values('Importance', ascending=False)

        self.metrics['feature_importance'] = importance
        return self.metrics

    def plot_results(self):
        """Plot actual vs predicted power output and residuals."""
        y_pred = self.model.predict(self.X_test_scaled)
        y_pred = np.clip(y_pred, a_min=0, a_max=None)

        plt.figure(figsize=(12, 6))
        plt.plot(self.y_test.index, self.y_test.values, label='Actual', alpha=0.8)
        plt.plot(self.y_test.index, y_pred, label='Predicted', alpha=0.8)
        plt.title('Hourly Solar Power Prediction (3 hours ahead)')
        plt.ylabel('Power Output (kW)')
        plt.xlabel('Time')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

        # Residual plot
        residuals = self.y_test - y_pred
        plt.figure(figsize=(12, 6))
        plt.scatter(y_pred, residuals, alpha=0.7)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.title('Residual Plot')
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.grid(True, alpha=0.3)
        plt.show()

    def run_pipeline(self):
        """Run the full pipeline: preprocessing, training, evaluating, and plotting."""
        self.preprocess_data()
        self.train_model()
        results = self.evaluate_model()
        self.plot_results()
        return results


# If using a CSV file:
# model = baseModel(file_path='preprocessed_data.csv')
# If using a pandas DataFrame:
# model = baseModel(data=df)

# results = model.run_pipeline()
