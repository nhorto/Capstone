import pandas as pd
import numpy as np

class SolarDataProcessor:
    import pandas as pd
    import numpy as np
    """
    A class for processing and combining solar inverter data with weather data.

    This class handles the preprocessing of inverter data by splitting it by inverter,
    filling missing values, and combining with weather data.
    """

    def __init__(self):
        """Initialize the SolarDataProcessor."""
        self.df = None
        self.weather_df = None
        self.combined_df = None

    def read_data(self, inverter_csv_path, weather_csv_path):
        """
        Read the inverter and weather data from CSV files.

        Parameters:
        inverter_csv_path (str): Path to the inverter CSV file
        weather_csv_path (str): Path to the weather CSV file
        """
        self.df = pd.read_csv(inverter_csv_path)
        self.weather_df = pd.read_csv(weather_csv_path)

        # Set pandas display options
        pd.set_option('display.max_columns', None)

        return self

    def split_inverter_data(self):
        """
        Split the inverter data by inverter number (1-3) and clean column names.
        """
        # Define additional columns to include in each inverter DataFrame
        additional_columns = ["Date", "Time"]

        # Initialize dictionary to store inverter columns
        inverter_columns = {1: [], 2: [], 3: []}

        # Identify columns for each inverter
        for col in self.df.columns:
            if "INVERTER 1" in col:
                inverter_columns[1].append(col)
            elif "INVERTER 2" in col:
                inverter_columns[2].append(col)
            elif "INVERTER 3" in col:
                inverter_columns[3].append(col)

        # Create DataFrames for each inverter and rename columns
        inverter_dfs = {}
        for inverter, cols in inverter_columns.items():
            # Ensure additional columns are included
            cols_with_additional = additional_columns + cols

            # Clean column names by removing the inverter identifier
            clean_cols = [col.replace(f"INVERTER {inverter}", "") for col in cols]

            # Keep the original names for additional columns
            clean_cols = additional_columns + clean_cols

            # Create a DataFrame for each inverter
            inverter_dfs[inverter] = self.df[cols_with_additional].copy()
            inverter_dfs[inverter].columns = clean_cols
            inverter_dfs[inverter].columns = inverter_dfs[inverter].columns.str.strip()

        one = inverter_dfs[1]
        two = inverter_dfs[2]
        three = inverter_dfs[3]

        return one, two, three

    def fill_missing_data(self, one, two, three):
        """
        Fill short gaps in solar inverter data using linear interpolation.
        """
        # Define the fill_missing_solar_data function
        def fill_missing_solar_data(df, columns):
            """
            Fill short gaps in solar inverter data using linear interpolation

            Parameters:
            df: DataFrame with datetime index
            columns: List of columns to process
            """
            df_filled = df.copy()

            for column in columns:
                if column not in df.columns:
                    print(f"Column '{column}' not found in DataFrame.")
                    continue

                # Simple linear interpolation for gaps up to 4 points
                df_filled[column] = df_filled[column].interpolate(
                    method='linear',
                    limit=4  # Only fill gaps of 4 or fewer points
                )
            return df_filled

        def interpolate_outliers(df, columns, factor=1.5):
          """
          Detect and replace outliers using IQR and linear interpolation.

          Parameters:
          - df: pd.DataFrame — Input DataFrame.
          - columns: list — Columns to check for outliers.
          - factor: float — IQR multiplier for determining outlier thresholds (default=1.5).

          Returns:
          - pd.DataFrame — DataFrame with outliers replaced via interpolation.
          """
          df_clean = df.copy()

          for col in columns:
              if col in ["Date", "Time"]:
                  continue

              Q1 = df_clean[col].quantile(0.25)
              Q3 = df_clean[col].quantile(0.75)
              IQR = Q3 - Q1
              lower_bound = Q1 - factor * IQR
              upper_bound = Q3 + factor * IQR

              # Mark outliers as NaN
              mask_outliers = (df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)
              df_clean.loc[mask_outliers, col] = np.nan

              # Interpolate to fill in outliers
              df_clean[col] = df_clean[col].interpolate(limit_direction='both')

          return df_clean


        cols1 = [col for col in one.columns if col not in ["Date", "Time"]]
        cols2 = [col for col in two.columns if col not in ["Date", "Time"]]
        cols3 = [col for col in three.columns if col not in ["Date", "Time"]]

        one = interpolate_outliers(one, cols1)
        two = interpolate_outliers(two, cols2)
        three = interpolate_outliers(three, cols3)

        one = fill_missing_solar_data(one, cols1)
        two = fill_missing_solar_data(two, cols2)
        three = fill_missing_solar_data(three, cols3)

        return one, two, three

    def combine_inverter_dfs(self, one, two, three):
        """
        Combine the three inverter DataFrames into a single DataFrame.
        """
        # Rename columns to include inverter number suffixes
        one = one.rename(columns={col: f"{col}_inv1" for col in one.columns if col not in ["Date", "Time"]})
        two = two.rename(columns={col: f"{col}_inv2" for col in two.columns if col not in ["Date", "Time"]})
        three = three.rename(columns={col: f"{col}_inv3" for col in three.columns if col not in ["Date", "Time"]})

        # Merge the three DataFrames on "Date" and "Time"
        df = one.merge(two, on=["Date", "Time"]).merge(three, on=["Date", "Time"])

        # Identify unique metric types by extracting column names without inverter numbers
        metric_types = set(col.rsplit("_", 1)[0] for col in df.columns if col not in ["Date", "Time"])

        # Aggregate metrics
        for metric in metric_types:
            # Select all inverter columns for this metric based on suffixes
            columns_to_combine = [col for col in df.columns if col.startswith(metric) and col not in ["Date", "Time"]]

            # Combine the columns (sum or mean, depending on metric type)
            if any(word in metric.lower() for word in ["power", "energy", "connections"]):
                df[f"total_{metric}"] = df[columns_to_combine].sum(axis=1)
            elif any(word in metric.lower() for word in ["temperature", "average", "percent"]):
                df[f"average_{metric}"] = df[columns_to_combine].mean(axis=1)
            else:
                df[f"total_{metric}"] = df[columns_to_combine].sum(axis=1)

            # Drop the original columns for this metric
            df.drop(columns=columns_to_combine, inplace=True)

        self.combined_df = df
        return self


    def add_datetime_and_hour(self):
        self.combined_df['Datetime'] = pd.to_datetime(self.combined_df['Date'] + ' ' + self.combined_df['Time'])
        self.combined_df['hour'] = self.combined_df['Datetime'].dt.hour

        self.weather_df['Datetime'] = pd.to_datetime(self.weather_df['Date'] + ' ' + self.weather_df['Time'])
        self.weather_df['hour'] = self.weather_df['Datetime'].dt.hour

        self.combined_df = self.combined_df.sort_values('Datetime')
        self.weather_df = self.weather_df.sort_values('Datetime')

        return self

    def merge_weather_data(self):
        self.combined_df = pd.merge_asof(self.combined_df, self.weather_df, on='Datetime', direction='nearest')

        self.combined_df = self.combined_df.drop(columns=['hour_x', 'Date_y', 'Time_y'])
        self.combined_df = self.combined_df.rename(columns={
            'Date_x':'date',
            'Time_x':'time',
            'hour_y':'hour'
        })
        return self

    def resample_time_series(self, data, interval='10min', tolerance='5min'):
        """
        Resample a time series to a consistent interval using closest real values when available,
        otherwise interpolate.

        Parameters:
        -----------
        data : pandas.DataFrame
            Original time series data with a 'Datetime' column or 'date' and 'time' columns.
        interval : str
            Desired resampling interval (e.g., '10min').
        tolerance : str
            Time window to search for a nearby real value (e.g., '2min').

        Returns:
        --------
        pandas.DataFrame
            Resampled DataFrame with consistent intervals using closest available data or interpolation.
        """

        df = data.copy()

        if 'Datetime' in df.columns:
            df['Datetime'] = pd.to_datetime(df['Datetime'])
        else:
            df['Datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'])

        df.set_index('Datetime', inplace=True)
        df.sort_index(inplace=True)

        original_index = df.index
        resample_index = pd.date_range(start=original_index.min(), end=original_index.max(), freq=interval)
        tolerance_offset = pd.to_timedelta(tolerance)

        # Initialize resampled DataFrame
        df_resampled = pd.DataFrame(index=resample_index)

        numeric_cols = df.select_dtypes(include='number').columns.tolist()

        for col in numeric_cols:
            resampled_col = []

            for target_time in resample_index:
                # Find the closest existing time within tolerance
                mask = (original_index >= target_time - tolerance_offset) & (original_index <= target_time + tolerance_offset)
                candidates = df.loc[mask, col]

                if not candidates.empty:
                    # Use the value closest in time
                    closest_time = candidates.index[np.argmin(np.abs(candidates.index - target_time))]
                    resampled_col.append(df.loc[closest_time, col])
                else:
                    resampled_col.append(np.nan)

            df_resampled[col] = resampled_col

            # Interpolate any remaining missing values
            df_resampled[col] = df_resampled[col].interpolate(method='time', limit_direction='both')

        # Final fill if needed 
        df_resampled = df_resampled.ffill().bfill()

        # Summary
        print(f"\nResampled to {interval} with tolerance {tolerance}")
        print(f"Resulting rows: {len(df_resampled)}")
        remaining_nans = df_resampled.isna().sum()
        if remaining_nans.sum() > 0:
            print("Warning: Columns with remaining NaNs:")
            print(remaining_nans[remaining_nans > 0])
        else:
            print("All missing values successfully filled.")

        self.combined_df = df_resampled
        return self


    def process_data(self, inverter_csv_path, weather_csv_path):
        """
        Process the data from start to finish in a single method call.
        ****** If you want to use the resample_time_series method, you will have to add to this method. I added it so we can get the timestamps at the same resolution
            it automatically updates the combined_df so all you should have to add to this method is self.resample_time_series()

        """
        self.read_data(inverter_csv_path, weather_csv_path)
        one, two, three = self.split_inverter_data()
        one, two, three = self.fill_missing_data(one, two, three)
        self.combine_inverter_dfs(one, two, three)
        self.add_datetime_and_hour()
        self.merge_weather_data()
        self.resample_time_series(data=self.combined_df)

        return self.combined_df 