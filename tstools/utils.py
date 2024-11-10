
def replace_nan_by_astronomical(df: pd.DataFrame | pd.Series, astronomical: pd.DataFrame | pd.Series) -> pd.DataFrame:
    """
    Replaces NaN values in a DataFrame or Series with corresponding values from another DataFrame or Series.

    This function will replace NaN values in the input `df` with values from the `astronomical` DataFrame or Series.
    If either `df` or `astronomical` is a Series, it will be converted to a DataFrame.

    Args:
        df (pd.DataFrame | pd.Series): The DataFrame or Series in which NaN values are to be replaced.
        astronomical (pd.DataFrame | pd.Series): The DataFrame or Series providing replacement values for NaNs in `df`.

    Returns:
        pd.DataFrame: A DataFrame with NaN values replaced by corresponding values from `astronomical`.
    """
    if type(df) == pd.Series:
        df = df.to_frame()
    if type(astronomical) == pd.Series: 
        astronomical = astronomical.to_frame()

    if len(df.columns) > 1:
        raise ValueError("Only DataFrames with one column are allowed")
    if len(astronomical.columns) > 1:
        raise ValueError("Only DataFrames (astronomical) with one column are allowed")

    df_column = df.columns[0]
    as_column = astronomical.columns[0]

    df = df.combine_first(astronomical.rename(columns={as_column: df_column}))
    
    return df

def replace_negative_by_nan(df: pd.DataFrame | pd.Series) -> pd.DataFrame:
    """
    Replaces negative values in a DataFrame or Series with NaN values.

    This function will replace all negative values in the input `df` with NaN values.
    If `df` is a Series, it will be converted to a DataFrame first.

    Args:
        df (pd.DataFrame | pd.Series): The DataFrame or Series in which negative values are to be replaced with NaNs.

    Returns:
        pd.DataFrame: A DataFrame with negative values replaced by NaNs.
    """
    if type(df) == pd.Series:
        df = df.to_frame()

    if len(df.columns) > 1:
        raise ValueError("Only DataFrames with one column are allowed")

    df_column = df.columns[0]
    
    negative = df[df_column] < 0
    
    df[negative] = np.nan
    
    return df

def find_largest_interval_without_nan(ds: pd.Series) -> pd.Series:
    """
    Finds the largest continuous interval in a pandas Series that does not contain any NaN values.

    This function takes a pandas Series as input and identifies the longest subsequence that does not
    include any NaN values. It returns this subsequence as a new Series.

    Args:
        ds (pd.Series): The input pandas Series in which to find the largest interval without NaN values.

    Returns:
        pd.Series: A pandas Series representing the largest continuous interval without NaN values.

    """
    a = ds.values
    m = np.concatenate(( [True], np.isnan(a), [True] ))  
    ss = np.flatnonzero(m[1:] != m[:-1]).reshape(-1,2)  
    start, stop = ss[(ss[:,1] - ss[:,0]).argmax()]  
    
    start = ds.index[start]
    stop = ds.index[stop-1]
    ds = ds.loc[start:stop]

    return ds

def astronomical_coefficients(ds: pd.Series, latitude: float) -> tuple[list[float], float]:
    """
    Perform tidal analysis on a given Series and calculate the astronomical coefficients.

    Args:
        ds (pd.Series): Input Series with a DateTime index.
        latitude (float): Latitude of the location for tidal analysis.

    Returns:
        tuple[list[float], float]: A list with the astronomical coefficients and the mean water level.
    """

    ds = find_largest_interval_without_nan(ds = ds)

    time = ds.index.values
    mean_level = np.mean(ds.values)
    levels = ds.values - mean_level

    coefs = utide.solve(time, levels, lat=latitude, verbose=False, method="ols")

    return coefs, mean_level

def tidal_prediction(coefs: dict, mean_level: float, predict_time: list[datetime]) -> pd.Series:
    """
    Predict tidal levels for a specified time period with help of astronomical coefficients. If no predict time is given the method will predict to the input data times.

    Args:
        coefs (list[float]): Astronomical coefficents.
        mean_level (float): The mean water level that will be added to the synthetic astronomical timeseries.
        predict_time (list[datetime]): Time values for which to predict tidal levels.

    Returns:
        pd.Series: Series with reconstructed astronomical tidal levels.
    """

    prediction = utide.reconstruct(predict_time, coefs, verbose=False)

    ds = pd.Series(prediction.h + mean_level, name=f"astronomical")
    ds.index = predict_time

    return ds

def tidal_analysis(ds, latitude, save_name, reference_time=None, predict_time=None, save_path=None):
    """
    Performs tidal analysis on the provided dataset and saves the results.

    Args:
        ds (pd.DataFrame): DataFrame containing the time series data with a datetime index.
        latitude (float): Latitude of the location for the tidal analysis.
        save_name (str): Base name for the saved files (coefficients and predictions).
        reference_time (list of str, optional): List with start and end dates for the reference period (e.g., ['2020-01-01', '2020-12-31']).
        predict_time (list of str, optional): List with start and end dates for the prediction period (e.g., ['2021-01-01', '2021-12-31']).
        save_path (str, optional): Directory path where the output files will be saved.

    Returns:
        None
    """
    if reference_time is None:
        reference_time = []
        reference_time.append(min(ds.index).strftime("%Y-%m-%d"))
        reference_time.append(max(ds.index).strftime("%Y-%m-%d"))

    coefs, mean_level = astronomical_coefficients(ds=ds.loc[reference_time[0]:reference_time[1]], latitude=latitude)
    
    coefs["meanWL"] = mean_level

    if predict_time is None:
        predict_time = reference_time

    wl = ds.loc[predict_time[0]:predict_time[1]]

    astronomical = tidal_prediction(predict_time=wl.index, coefs=coefs, mean_level=coefs["meanWL"])

    if save_path is None:
        save_path = Path(".")
    else:
        save_path = Path(save_path).parent
    
    file_coef = f"{save_path}\\{save_name}_coef.pkl"
    file_pred = f"{save_path}\\{save_name}_pred.csv"

    with open(file_coef, 'wb') as f:
        pickle.dump(coefs, f)

    astronomical.to_frame().to_csv(file_pred, sep=";")

def polar_to_cartesian(rho: float, phi: float) -> tuple[float, float]:
    """
    Converts polar coordinates to Cartesian coordinates. 

    Args:
        rho (float): The distance from the origin.
        phi (float): The angle in radians.

    Returns:
        tuple(float, float): The x and y coordinates of the point in Cartesian
            coordinates.
    """

    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    
    return(x, y)

def wind_speed_direction_to_u_v(df: pd.DataFrame, wind_speed_name: str, wind_direction_name: str, u_name: str, v_name: str):
    """
    Converts wind speed and direction to u and v components.

    Args:
        df (pd.DataFrame): A pandas DataFrame containing wind speed and direction data.
        wind_speed_name (str): The name of the column containing wind speed data.
        wind_direction_name (str): The name of the column containing wind direction data.
        u_name (str): The name of the column for the u component.
        v_name (str): The name of the column for the v component.

    Returns:
        pd.DataFrame: A pandas DataFrame with wind speed and direction columns replaced by u and v components.
    """
    u, v = polar_to_cartesian(df[wind_speed_name], df[wind_direction_name])
    df[u_name] = u
    df[v_name] = v
    df = df.drop(columns=[
        wind_speed_name,
        wind_direction_name,
        ])
    return df


class PlotterNanRange:
    def __init__(self):
        """
        Initializes the PlotterNanRange class.
        """
        self.df_list = []
        self.legends = []
        self.consecutive_nans_list = []

    def add_data(self, df: pd.DataFrame, label: str = None, consecutive_nans: int = 1):
        """
        Adds a DataFrame to the plotter.

        Args:
            df (pd.DataFrame): The DataFrame containing NaN values to plot. The DataFrame must have a datetime index.
            label (str, optional): The name of the DataFrame to display in the plot legend. Defaults to None.
            consecutive_nans (int): Minimum number of consecutive NaNs to plot. Defaults to 1.
        """
        self.df_list.append(df)
        self.consecutive_nans_list.append(consecutive_nans)
        
        if label is not None:
            self.legends.append(label)
        else:
            self.legends.append(f"DataFrame {len(self.df_list)}")

    def plot(self, legend: bool = False, xlabel: str = 'CET (UTC+1)', title = 'NanRanges'):
        """
        Plots the ranges of NaN values for the added DataFrames.

        Args: 
            legend (bool): Display a legend or not. Defaults to False.
            xlabel (str): Name of the x label of the plot. Defaults to 'CET (UTC+1)'.
            title (str): Name of the title of the plot. Defaults to 'NanRanges'.
        """
        if not self.df_list:
            raise ValueError("No dataframes added. Please add dataframes using the add_data method before plotting.")

        fig, ax = plt.subplots()

        colors = plt.cm.Set2(np.linspace(0, 1, len(self.df_list)))

        y = 0
        y_labels = []
        legend_dummy = []

        for ii, df_i in enumerate(self.df_list):
            consecutive_nans = self.consecutive_nans_list[ii]
            for column in df_i.columns:
                is_nan = df_i[column].isna()
                is_nan = self._consecutive_trues(is_nan, consecutive_nans)
                y_list = np.ones(sum(is_nan)) * y
                if sum(is_nan) == 0:
                    ax.plot(df_i.index, np.ones(len(df_i)) * y, 'o', color="White", markersize=2, zorder=-100)
                else:
                    ax.plot(df_i[is_nan].index, y_list, 'o', color=colors[ii], markersize=2)
                y += 1
                y_labels.append(column)
            if legend:
                legend_dummy.append(Line2D([0], [0], label=self.legends[ii], marker='s', markersize=5, 
                                       markeredgecolor=colors[ii], markerfacecolor=colors[ii], linestyle=''))

        ax.set_yticks(range(len(y_labels)))
        ax.set_yticklabels(y_labels)

        min_date = min([min(df_i.index) for df_i in self.df_list])
        max_date = max([max(df_i.index) for df_i in self.df_list])

        ax.invert_yaxis()
        if legend:
            plt.legend(legend_dummy, loc='center left')
        plt.xlabel(xlabel)
        plt.title(title)
        plt.xlim(min_date, max_date)
        plt.show()

    @staticmethod
    def _consecutive_trues(arr: list, n: int):
        """
        Finds ranges of consecutive True values in a boolean array.

        Args:
            arr (list): The boolean array.
            n (int): Minimum number of consecutive True values.

        Returns:
            np.ndarray: A boolean array indicating ranges of consecutive True values.
        """
        arr = np.array(arr)
        rolling_view = np.lib.stride_tricks.sliding_window_view(arr, n)
        mask = np.all(rolling_view, axis=-1)
        result = np.zeros_like(arr, dtype=bool)
        for i in np.where(mask)[0]:
            result[i:i+n] = True
        return result
    
    
class PlotterDataAvailability:
    def __init__(self):
        """
        Initializes the PlotterDataAvailability class.
        """

        self.timeseries = []
        self.colors = plt.cm.Set2(np.linspace(0, 1, 8))


    def add_data(self, df: pd.DataFrame, label: str = None, group: str = None, consecutive_nans: int = 1):
        """
        Adds a DataFrame to the plotter.

        Args:
            df (pd.DataFrame): The DataFrame containing NaN values to plot.
            label (str, optional): The name of the DataFrame to display in the plot legend. Defaults to None.
            group (str, optional): The name of the DataFrame to display in the plot legend and by which the colors are grouped. Defaults to None.
            consecutive_nans (int, optional): Minimum number of consecutive NaNs to plot. Defaults to 1.

        Returns:
            None
        """
        ## TODO: Add support for multiple columns
        assert df.shape[1] == 1, f"DataFrame has {df.shape[1]} columns, but currently it should have exactly 1."

        if not self.timeseries:
            color = self.colors[0]
        else:
            groups = [ts.group for ts in self.timeseries]
            colors = [ts.color for ts in self.timeseries]
            n_groups = len((np.unique(groups)))
            if group in groups:
                idx = groups.index(group)
                color = colors[idx]
            else: 
                color = self.colors[n_groups]

        if label is None:
            label = df.columns[0]

        self.timeseries.append(TimeSeriesData(df=df, group=group, consecutive_nans=consecutive_nans, color=color, label = label))

    def plot(self, legend: bool = False, xlabel: str = 'UTC', title = 'Data Availability', nan_ranges = False):
        """
        Plots the data availability for the added dataframes.

        Args:
            legend (bool, optional): Whether to display a legend. Defaults to False.
            xlabel (str, optional): The label for the x-axis. Defaults to 'UTC'.
            title (str, optional): The title of the plot. Defaults to 'Data Availability'.
            nan_ranges (bool, optional): Whether to plot nan ranges or available data ranges. Defaults to False.

        Returns:
            None
        """
        if not self.timeseries:
            raise ValueError("No timeseries dataframes added. Please add dataframes using the add_data method before plotting.")

        self.fig, self.ax = plt.subplots()

        y = 0
        y_labels = []
        legend_dummy = []

        for ii, ts in enumerate(self.timeseries):
            is_nan = ts.df[ts.df.columns[0]].isna()
            is_nan = self._consecutive_trues(is_nan, ts.consecutive_nans)
            if nan_ranges:
                avail_bool = is_nan
            else:
                avail_bool = ~is_nan
            y_list = np.ones(sum(avail_bool)) * y
            if sum(avail_bool) == 0:
                self.ax.plot(ts.df.index, np.ones(len(ts.df)) * y, 'o', color="White", markersize=2, zorder=-100)
            else:
                self.ax.plot(ts.df[avail_bool].index, y_list, 'o', color="gray", markersize=2)
            y_labels.append((ts.label, ts.color))
            y += 1

        self.ax.set_yticks(range(len(y_labels)))
        self.ax.set_yticklabels([ylabel for ylabel, _ in y_labels])
        for tick_label, (_, color) in zip(self.ax.get_yticklabels(), y_labels):
            tick_label.set_color(color)

        min_date = min([min(ts.df.index) for ts in self.timeseries])
        max_date = max([max(ts.df.index) for ts in self.timeseries])

        self.ax.invert_yaxis()
        plt.tight_layout()
    
        if legend:
            groups = [ts.group for ts in self.timeseries]
            colors = [ts.color for ts in self.timeseries]
            unique_labels = np.unique(groups)
            for label in unique_labels:
                idx = groups.index(label)
                color = colors[idx] 
                legend_dummy.append(Line2D([0], [0], label=label, marker='s', markersize=5, 
                                    markeredgecolor=color, markerfacecolor=color, linestyle=''))
            self.fig.legend(handles=legend_dummy, loc='center left', bbox_to_anchor=(1, 0.5))
        plt.xlabel(xlabel)
        plt.title(title)
        plt.xlim(min_date, max_date)
        plt.show

    def xlim(self, min_date: str, max_date: str):
        """
        Sets the x-axis limits.

        Args:
            min_date (str): The minimum date to show in the x-axis.
            max_date (str): The maximum date to show in the x-axis.

        Returns:
            None
        """
        self.ax.set_xlim([pd.to_datetime(min_date), pd.to_datetime(max_date)])    

    def show(self):
        """
        Displays the current plot.

        Args:
            None

        Returns:
            None
        """
        plt.show()

    @staticmethod
    def _consecutive_trues(arr: list, n: int):
        """
        Finds ranges of consecutive True values in a boolean array.

        Args:
            arr (list): The boolean array.
            n (int): Minimum number of consecutive True values.

        Returns:
            np.ndarray: A boolean array indicating ranges of consecutive True values.
        """
        arr = np.array(arr)
        rolling_view = np.lib.stride_tricks.sliding_window_view(arr, n)
        mask = np.all(rolling_view, axis=-1)
        result = np.zeros_like(arr, dtype=bool)
        for i in np.where(mask)[0]:
            result[i:i+n] = True
        return result
