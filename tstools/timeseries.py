"""
Module Name: Timeseries Tools
Author: Faro Schäfer
Date: July 3, 2024

Description:
The timeseries module contains the timeseries class
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from mikeio import EUMType, EUMUnit
import pytz

from converter import to_dfs0

class TimeSeries:
    def __init__(self, df: pd.DataFrame, name: str, timezone: str, type: EUMType | None = None, unit: EUMUnit | None = None) -> None:
        """
        Initializes a TimeSeries object.

        Args:
            df (pd.DataFrame): The input DataFrame.
            name (str): The name of the TimeSeries.
            timezone (str): The time zone or fixed offset (e.g., 'America/Chicago' or 'UTC-6').
            type (str): The type of the TimeSeries (e.g. 'Surface Elevation' or 'Current Speed')
            unit (str): The unit of the TimeSeries (e.g. 'meter' or 'meter_per_second').

        Returns:
            None
        """
        if not pd.api.types.is_datetime64_any_dtype(df.index):
            raise TypeError("The DataFrame index must be of datetime type.")

        self.df = df
        self.label = name
        self.type = type
        self.unit = unit
        self.timezone = timezone

        self._localize_time_index(timezone)
        self.df = self.df.sort_index()

        self._prepare_columns()

    def __copy__(self):
        return TimeSeries(df=self.df, name=self.label, type=self.type, unit=self.unit, timezone=self.timezone)

    def combine(self, ts: "TimeSeries", keep: str | None = None) -> "TimeSeries":
        if self.df.index.tz != ts.df.index.tz:
                raise ValueError(f"The timezones of the TimeSeries objects do not match ({self.df.index.tz} and {ts.df.index.tz})")

        overlap_start = max(self.df.index.min(), ts.df.index.min())
        overlap_end = min(self.df.index.max(), ts.df.index.max())
        
        if overlap_start <= overlap_end:
            if keep == None:
                raise ValueError("Time periods overlap please specify keep = 'first' or 'second' to choose which data to keep.")
            elif keep == 'first':
                df = self.df.combine_first(ts.df[~ts.df.index.isin(self.df.index)])
            elif keep == 'second':
                df = self.df[~self.df.index.isin(ts.df.index)].combine_first(ts.df)
        else:
            df = self.df.combine_first(ts.df)

        return TimeSeries(df = df, name = f"{self.label}_{ts.label}", type = self.type, unit = self.unit, timezone = self.timezone)

    def resample(self, resample_interval: str) -> "TimeSeries":
        """
        Returns the resampled TimeSeries.

        Args:
            resample_interval (str): The interval to resample the DataFrame by.

        Returns:
            TimeSeries: The resampled TimeSeries.
        """
        df = self.df.resample(resample_interval).mean()
        df.index.freq = self.df.index.inferred_freq

        return TimeSeries(df=df, name=self.label, type=self.type, unit=self.unit, timezone=self.timezone)

    def float_to_nan(self, to_nan: float):
        """
        Replaces all occurrences of a specified float value with NaN in the DataFrame.

        Args:
            to_nan (float): The float value to be replaced with NaN.

        Returns:
            None
        """
        self.df = self.df.replace(to_nan, np.nan)   
            
    def find_gaps(self, gap_interval: str = "1D") -> pd.DataFrame:
        """
        Finds gaps larger than the specified gap interval in the DataFrame's datetime index,
        including gaps caused by NaN values in the DataFrame.

        Args:
            gap_interval (str): The minimum duration of gaps to find, expressed as a pandas 
                                offset alias string (e.g., "1D" for 1 day, "2H" for 2 hours). 
                                Default is "1D".

        Returns:
            pd.DataFrame: A DataFrame containing the start and end of each gap larger than the 
                          specified interval, and the duration of the gap. The DataFrame has 
                          columns:
                          - 'Start': The start of the gap.
                          - 'End': The end of the gap.
                          - 'Gap Duration': The duration of the gap.
        """
        nan_indices = self.df[self.df.isna().any(axis=1)].index
        combined_indices = self.df.index.append(nan_indices).sort_values()
        time_differences = combined_indices.to_series().diff()
        gaps = time_differences[time_differences > pd.Timedelta(gap_interval)]

        gap_info = pd.DataFrame({
            'Start': gaps.index - gaps.values,
            'End': gaps.index,
            'Duration': gaps.values
        })

        return gap_info
    
    def find_availability(self, gap_interval: str = "1D") -> pd.DataFrame:
        """
        Finds available intervals in the DataFrame's datetime index,
        excluding gaps larger than the specified gap interval.

        Args:
            gap_interval (str): The minimum duration of gaps to consider, expressed as a pandas 
                                offset alias string (e.g., "1D" for 1 day, "2H" for 2 hours). 
                                Default is "1D".

        Returns:
            pd.DataFrame: A DataFrame containing the start and end of each available interval 
                        between gaps larger than the specified interval, and the duration of 
                        the interval. The DataFrame has columns:
                        - 'Start': The start of the available interval.
                        - 'End': The end of the available interval.
                        - 'Available Duration': The duration of the available interval.
        """
        gap_info = self.find_gaps(gap_interval)

        available_starts = list(gap_info['Start'])
        available_starts.insert(0, self.df.index[0])

        available_ends = list(gap_info['End'])
        available_ends.append(self.df.index[-1])

        available_durations = np.array(available_ends) - np.array(available_starts)
        
        avail_info = pd.DataFrame({
            'Start': available_starts,
            'End': available_ends,
            'Duration': available_durations
        })

        return avail_info
    
    def convert_timezone(self, timezone: str) -> None:
        """
        Converts the timezone of the DataFrame to the specified timezone.

        Args:
            timezone (str): The timezone to convert the DataFrame to.

        Returns:
            None
        """
        self.timezone = timezone
        try:
            self.df.index = self.df.index.tz_convert(timezone)
        except pytz.UnknownTimeZoneError:
            raise ValueError(f"Unknown time zone: {timezone}")

    ##### I/O

    def to_dfs0(self, save_file: str) -> None:
        """
        Convert the timeseries data to a .dfs0 file.

        Args:
            save_file (str): The path to save the .dfs0 file.

        Returns:
            None
        """
        to_dfs0(variables = self.df[self.column_name].to_numpy(), 
                time = self.df.index.to_numpy(),
                variable_names=self.column_name, 
                eumtypes=self.type, 
                eumunits=self.unit, 
                save_file=save_file)
        
    ##### Internal 

    def _prepare_columns(self) -> None:
        """
    	Prepares the columns of the DataFrame.

    	Sets the column name of the DataFrame to a formatted string that includes
    	the display name of the type and the short name of the unit.

    	Args:
    		None

    	Returns:
    		None
    	"""

        self.df.index.name = "Timestamp"
        self.column_name = f"{self.type.display_name} [{self.unit.short_name}]"
        self.df.columns = [self.column_name]

    def _localize_time_index(self, timezone: str) -> None:
        """
        Localizes the DataFrame's index to the given timezone or fixed offset.

        Args:
            timezone (str): The timezone or UTC offset to localize to.

        Returns:
            None
        """
        if timezone.startswith('UTC') and '-' in timezone or '+' in timezone:
            # Handling for fixed offsets like 'UTC-6'
            try:
                offset_hours = int(timezone.split('UTC')[1])
                fixed_offset = pytz.FixedOffset(offset_hours * 60)
                self.df.index = self.df.index.tz_localize(fixed_offset)
            except ValueError:
                raise ValueError("Invalid UTC offset format. Use 'UTC±X' where X is an integer.")
        else:
            # Handling for named time zones (e.g., 'America/Chicago')
            try:
                self.df.index = self.df.index.tz_localize(timezone, nonexistent='shift_forward')
            except pytz.UnknownTimeZoneError:
                raise ValueError(f"Unknown time zone: {timezone}")
            
    ##### Plot 

    def plot(self)-> None:
        """
        Plot the time series.

        Plots the time series data with a line plot. The x-axis is the time and the y-axis is the value of the time series.

        Args:
            None

        Returns:
            None
        """
        self.df.plot(figsize=(20, 10))

## TODO: Add support for multiple time series. Current class is not working. 
class MultiTimeSeries():
    timeseries: list[TimeSeries] = []

    def __init__(self, timeseries: list[TimeSeries]):
        """
        Initialize a MultiTimeSeries object.

        Args:
            timeseries (list[TimeSeries]): A list of TimeSeries objects.
        """

        self.df = pd.DataFrame()
        for ts in timeseries:
            self.add_timeseries(ts)

    def add_timeseries(self, timeseries: TimeSeries):
        """
        Adds a TimeSeries to the MultiTimeSeries object.

        Args:
            timeseries (TimeSeries): The TimeSeries object to add.

        Returns:
            None
        """
        
        self._check_validity()
                    
        self.df = pd.concat([self.df, timeseries.df], axis=1)
        timeseries.df = self.df[timeseries.df.columns]
        self.timeseries.append(timeseries)
        self.df.columns = [ts.label + ts.column_name for ts in self.timeseries]

        self.time = self.df.index

    def _check_validity(self) -> pd.DataFrame:
        
        if not self.timeseries:
            pass
        else:
            if not all([ts.df.index.tz == self.timeseries[0].df.index.tz for ts in self.timeseries]):
                raise ValueError("The timezones of the TimeSeries objects do not match. Please convert them to the same timezone before creating a MultiTimeSeries.")
        
    def to_dfs0(self, save_file: str) -> None:
        """
        Convert the timeseries data to a .dfs0 file.

        Args:
            save_file (str): Path of the save file. If None the file will be named like the .netcdf file. Defaults to None.

        Returns:
            None
        """
        converter = DataFrameConverter(self.df)

        converter.to_dfs0(variable_names = self.df.columns, 
                           eumtypes = [ts.type for ts in self.timeseries],
                           eumunits = [ts.unit for ts in self.timeseries], 
                           save_file=save_file)