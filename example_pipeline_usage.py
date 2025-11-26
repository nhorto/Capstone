"""
Example Usage for run_pipeline.ipynb

This file contains example code showing how to use the preprocessing,
individual models, and blender ensemble model.

Copy and paste the relevant sections into your Jupyter notebook cells.
"""

# =============================================================================
# CELL 1: Imports
# =============================================================================
import sys
import os
sys.path.append(os.getcwd())  # Add current directory to Python path

import pandas as pd
from preprocessing import SolarDataProcessor
from base_model import baseModel
from XGBoost import XGBoost
from LSTM import LSTMForecastModel
from Blender import blender


# =============================================================================
# CELL 2: Preprocess Your Raw Data
# =============================================================================
processor = SolarDataProcessor()
df = processor.process_data('Inv_2024-09-30.csv', 'weather_2024-09-30.csv')

print(f"Preprocessed data shape: {df.shape}")
print(f"Date range: {df.index.min()} to {df.index.max()}")


# =============================================================================
# CELL 3 (OPTION A): Run Base Model Only
# =============================================================================
print("\n" + "="*50)
print("RUNNING BASE MODEL")
print("="*50)

base_model = baseModel(data=df)
base_model_results = base_model.run_pipeline()

print("\nBase Model Results:")
print(f"RMSE: {base_model_results['rmse']:.3f}")
print(f"MAE: {base_model_results['mae']:.3f}")
print(f"R²: {base_model_results['r2']:.3f}")


# =============================================================================
# CELL 4 (OPTION B): Run XGBoost Model Only
# =============================================================================
print("\n" + "="*50)
print("RUNNING XGBOOST MODEL")
print("="*50)

xgb_model = XGBoost(data=df.reset_index())

# Option 1: Run full pipeline (includes feature analysis, training, evaluation, plotting)
xgb_results = xgb_model.run_pipeline(
    target_horizon=18,        # Predict 18 steps ahead (3 hours with 10-min data)
    resample_freq='30T',      # Resample to 30-minute intervals
    use_top_features=True,    # Use only top N features (recommended)
    top_n=30                  # Number of top features to use
)

print("\nXGBoost Results:")
print(f"RMSE: {xgb_results['test_metrics']['rmse']:.3f}")
print(f"MAE: {xgb_results['test_metrics']['mae']:.3f}")
print(f"R²: {xgb_results['test_metrics']['r2']:.3f}")

# Option 2: Run step-by-step (for more control)
# xgb_model.preprocess_data(target_horizon=18, resample_freq='30T')
# xgb_model.train_model()
# train_metrics, test_metrics = xgb_model.evaluate_model()
# xgb_model.plot_results()


# =============================================================================
# CELL 5 (OPTION C): Run LSTM Model Only
# =============================================================================
print("\n" + "="*50)
print("RUNNING LSTM MODEL")
print("="*50)

lstm_model = LSTMForecastModel(
    data=df,
    lookback=5  # Number of timesteps to look back (adjust based on resample_freq)
)

# Run full pipeline
lstm_results = lstm_model.run_pipeline(
    epochs=100,
    batch_size=64
)

print("\nLSTM Results:")
print(f"RMSE: {lstm_results['rmse']:.3f}")
print(f"MAE: {lstm_results['mae']:.3f}")
print(f"R²: {lstm_results['r2']:.3f}")


# =============================================================================
# CELL 6 (OPTION D): Run Blended Ensemble Model (RECOMMENDED)
# =============================================================================
print("\n" + "="*50)
print("RUNNING BLENDED ENSEMBLE MODEL")
print("="*50)

# Create blender instance
blend = blender(
    data=df,
    train_ratio=0.8,   # 80% for training base models
    val_ratio=0.1,     # 10% for training meta-model
    test_ratio=0.1     # 10% for final evaluation
)

# Run both base models (LSTM and XGBoost)
blend.run_models(
    lstm_freq='15min',      # LSTM works better with higher resolution
    xgb_freq='60min',       # XGBoost can handle coarser resolution
    lstm_horizon=18,        # Predict 18 steps ahead
    xgb_horizon=18          # Predict 18 steps ahead
)

# Train the meta-model (blends LSTM and XGBoost predictions)
blend_predictions = blend.train_blended_model()

# Train residual correction model (optional but recommended)
blend.train_residual_model()
blend.apply_residual_correction()

# Evaluate the final blended model
mae, rmse, r2 = blend.evaluate_blend_model(blend.y_blend_true)

print("\nBlended Model Results:")
print(f"MAE: {mae:.3f}")
print(f"RMSE: {rmse:.3f}")
print(f"R²: {r2:.3f}")

# Plot diagnostics (if method exists)
# blend.plot_blend_diagnostics()


# =============================================================================
# CELL 7: Compare All Models (Optional)
# =============================================================================
print("\n" + "="*50)
print("MODEL COMPARISON")
print("="*50)

comparison = pd.DataFrame({
    'Model': ['Base (Linear Reg)', 'XGBoost', 'LSTM', 'Blended Ensemble'],
    'RMSE': [
        base_model_results['rmse'],
        xgb_results['test_metrics']['rmse'],
        lstm_results['rmse'],
        rmse
    ],
    'MAE': [
        base_model_results['mae'],
        xgb_results['test_metrics']['mae'],
        lstm_results['mae'],
        mae
    ],
    'R²': [
        base_model_results['r2'],
        xgb_results['test_metrics']['r2'],
        lstm_results['r2'],
        r2
    ]
})

print(comparison.to_string(index=False))


# =============================================================================
# CELL 8: Making Predictions on New Data (Optional)
# =============================================================================
print("\n" + "="*50)
print("MAKING PREDICTIONS ON NEW DATA")
print("="*50)

# Example: Load new data and make predictions
# new_data = pd.read_csv('new_inverter_data.csv')

# Using XGBoost model
# xgb_predictions = xgb_model.predict(data=new_data)

# Using LSTM model
# lstm_predictions = lstm_model.predict(
#     new_data=new_data,
#     resample_freq='30min',
#     target_horizon=18
# )

print("See commented code above for prediction examples")


# =============================================================================
# NOTES AND TIPS
# =============================================================================
"""
IMPORTANT NOTES:

1. Data Requirements:
   - Your CSV files must contain 'Date' and 'Time' columns
   - Or a single 'datetime' or 'index' column with datetime values
   - Weather data should include 'WMS 01 irradiance' and 'average_temperature_C'
   - Inverter data should include power output columns

2. Hyperparameter Tuning:
   - LSTM lookback: Adjust based on your resampling frequency
     * 15-min intervals: lookback=5-10
     * 30-min intervals: lookback=3-5
     * 60-min intervals: lookback=2-3

   - XGBoost resample_freq:
     * Smaller intervals (10-30min) capture more detail but slower training
     * Larger intervals (60min) faster training but less detail

   - target_horizon: Number of steps ahead to predict
     * For 10-min data, 18 steps = 3 hours ahead
     * Adjust based on your use case

3. Feature Engineering:
   - Both LSTM and XGBoost now use the same features (as of latest update)
   - Features include: lags, rolling statistics, differences, interactions
   - See feature_engineering_plan in each model class for details

4. Model Selection:
   - Base Model: Fast, simple, good baseline
   - XGBoost: Best for most cases, handles non-linearity well
   - LSTM: Good for capturing temporal patterns, requires more data
   - Blended: Best performance, combines strengths of all models

5. Performance:
   - LSTM training is GPU-accelerated if TensorFlow GPU is available
   - XGBoost can use all CPU cores (n_jobs=-1)
   - Blender runs both models so takes longest

6. Common Issues:
   - If you get "not enough data" errors, reduce lookback or target_horizon
   - If predictions are all zeros, check for data leakage or scaling issues
   - If RMSE is very high, check your target_horizon aligns with your data frequency
"""
