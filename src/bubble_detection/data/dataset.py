from typing import Dict, List
from datetime import timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf


class StockDataset:
    def __init__(self, ticker):
        self.ticker = yf.Ticker(ticker)
        self.hist = None
        self.drawdowns = None

    def get_hist(self, start, end, interval='1d'):
        print(f"=== LOAD {self.ticker.ticker} data from {start} to {end} by {interval} ===")
        df = self.ticker.history(start=start, end=end, interval=interval)
        if df.shape[0] == 0:
            raise ValueError("No data found, hourly data range must be within the last 730 days.")
        # Remove timezone
        df.index = df.index.tz_localize(None)
        df.index.name = df.index.name.lower()
        df.columns = df.columns.str.lower()

        df = df[['open', 'high', 'low', 'close', 'volume']]
        # df['diff'] = df['close'].diff().fillna(0)
        # df['return'] = df['close'].pct_change().fillna(0)
        df['diff'] = df['close'].diff()
        df['return'] = df['close'].pct_change()
        df['vol'] = df['return'].rolling(10).std()
        df = df.dropna()
        df.index = pd.to_datetime(df.index)
        self.hist = df
        return df.copy(deep=True)

    def get_change(self, is_up: bool):
        """
        Get all change period dataframe given up or down.
        Args:
            is_up: If True then choose all drawup periods, else then choose all drawdown periods.

        Returns: Dataframe with all either drawup or drawdown.

        """
        pmin_pmax = (self.hist['close'].diff(-1) > 0).astype(int).diff()
        change_point = pmin_pmax[pmin_pmax != 0].index

        change_df = self.hist.loc[change_point, 'close'].reset_index()
        change_df['start_date'] = change_df['date'].shift(1)
        change_df['return'] = change_df['close'].pct_change()
        change_df.rename(columns={'date': 'end_date'}, inplace=True)
        change_df = change_df[['start_date', 'end_date', 'close', 'return']].drop(index=0)
        change_df['duration'] = pd.to_datetime(change_df['end_date']) - pd.to_datetime(change_df['start_date'])

        if is_up:
            return change_df[change_df['return'] >= 0].reset_index(drop=True)
        else:
            return change_df[change_df['return'] < 0].reset_index(drop=True)

    def get_massive_change(self, is_up: bool, method, pct=0.05, past_period=120):
        """
        Filter out massive changes among all change periods by quantile.

        Args:
            is_up: If True then choose drawup, else then choose drawdown periods.
            method: The way to define a massive change. If the change return falls outside of give quantile, then it's massive.
                Options:
                    'all': Use all history return as distribution.
                    'past_all': Use all history return before corresponding date as distribution.
                    'past_period': Use past_period history return before corresponding date as distribution.
            pct: The quantile threshold.
            past_period: number of time steps to lookback for past_period method.

        Returns: Dataframe with either massive drawup or massive drawdown.

        """
        change_df = self.get_change(is_up)
        abs_return = change_df['return'].abs()
        if method == 'all':
            change_df = change_df[abs_return >= abs_return.quantile(1 - pct)]
        elif method == 'past_all':
            change_df = change_df[abs_return >= abs_return.expanding().quantile(1 - pct)]
        elif method == 'past_period':
            change_df = change_df[abs_return >= abs_return.rolling(window=past_period, min_periods=1).quantile(1 - pct)]
        return change_df.reset_index(drop=True)

    def get_change_forecast_label(self, forecast_len: int, **kwargs):
        """
        Given massive change start date, mark previous `forecast_days` trade days as 1.
        Thus a date label is 1, indicating there will be a massive change(drawup or drawdown) in `forecast_days`.

        Args:
            forecast_len: The number of trade days to forecast massive ahead.
            **kwargs: parameters of self.get_massive_change()

        Returns: Binary label series.

        """
        massive_change = self.get_massive_change(**kwargs)
        labels = pd.Series(0, index=self.hist.index)
        for date in massive_change['start_date']:
            iloc = labels.index.get_loc(date)
            if iloc < forecast_len:
                labels[:iloc] = 1
            else:
                labels[iloc - forecast_len:iloc] = 1
        return labels

    def lookback_agg(self, lookback_len, agg_func: Dict = None, new_col_name: List = None):
        """
        Apply aggregation functions on lookback period.

        Args:
            lookback_len: The number of trade days to lookback.
            agg_func: Aggregation functions, in structure {'column name': ['function to apply']}.
            new_col_name: List of column names of aggregated functions.

        Returns: Lookback aggregated dataframe.

        """
        if agg_func is None:
            agg_func = {
                'high': max,
                'low': min,
                'close': [np.mean, np.std],
                'diff': [np.mean, np.std],
                'return': [np.mean, np.std],
                'volume': [np.mean, np.std],
            }
            new_col_name = [
                f'past_{lookback_len}_max', f'past_{lookback_len}_min',
                f'past_{lookback_len}_avg', f'past_{lookback_len}_std',
                f'past_{lookback_len}_diff_avg', f'past_{lookback_len}_diff_std',
                f'past_{lookback_len}_return_avg', f'past_{lookback_len}_return_std',
                f'past_{lookback_len}_volume_avg', f'past_{lookback_len}_volume_std',
            ]
        agg_df = self.hist.rolling(lookback_len).agg(agg_func)
        agg_df.columns = new_col_name
        return agg_df

    def get_drawdowns(self, df=None, method='fixed', window_len=21, drop_min=0.15,
                      e0_start=0.1, e0_end=5, e0_step=0.1, w_start=10, w_end=60, w_step=5, freq_min=0.5):
        if df is None:
            df = self.hist
        if method == 'fixed':
            self.drawdowns = get_fixed_drawdowns(df, window_len, drop_min)
        elif method == 'dynamic':
            self.drawdowns = get_dynamic_drawdowns(
                df, e0_start, e0_end, e0_step, w_start, w_end, w_step, freq_min
            )
        else:
            raise ValueError(f'method {method} not supported')
        return self.drawdowns

    def shuffle_df(self, df, seed, **drawdowns_kwargs):
        shuffled_df = pd.DataFrame(index=df.index)
        sheffled_return = df['return'][1:].sample(frac=1.0, random_state=seed)
        shuffled_price = [df['close'][0]]
        for r in sheffled_return:
            shuffled_price.append(shuffled_price[-1] * (r + 1))
        shuffled_df['close'] = shuffled_price
        shuffled_df['return'] = [df['return'][0]] + sheffled_return.tolist()
        shuffled_drawdowns = self.get_drawdowns(df=shuffled_df, **drawdowns_kwargs)
        print(f"{shuffled_drawdowns.shape[0]} drawdowns found in shuffle")
        return shuffled_df, shuffled_drawdowns

    def generate_features(self, win: int = 40):
        self.hist['return_vol'] = self.hist['return'].rolling(10).std()

        # moving average
        ma_day = np.arange(5, win, 5)
        moving_features = {}
        for ma in ma_day:
            close_column_name = f"close_MA_for_{ma}_days"
            moving_features[close_column_name] = self.hist['close'].rolling(ma).mean()

            return_column_name = f"return_MA_for_{ma}_days"
            moving_features[return_column_name] = self.hist['return'].rolling(ma).mean()

            vol_column_name = f"vol_MA_for_{ma}_days"
            moving_features[vol_column_name] = self.hist['return_vol'].rolling(ma).mean()
        moving_features = pd.DataFrame.from_dict(moving_features)

        return moving_features


def get_fixed_drawdowns(df, window_len=21, drop_min=0.15, delta='day'):
    """
    Generate drawdowns that occurred in fixed length of windows and had fixed drop percentage.

    Args:
        df: the dataframe of stock price
        window_len: length of window
        drop_min: threshold of drop percentage

    Returns: a dataframe with 5 columns

    """

    j = 0
    res = []
    while j < df.shape[0] - window_len:
        x = df.iloc[j]['close']
        valley = x
        valley_idx = 0
        for i in range(1, window_len):
            p = df.iloc[j + i]['close']
            if p < x:
                if p < valley:
                    valley = p
                    valley_idx = i
            else:
                break
        drop = (x - valley) / x
        #         peak_date = datetime.strptime(df.index[j], "%Y-%m-%d")
        peak_date = df.index[j]
        if delta == 'day':
            valley_date = peak_date + timedelta(days=valley_idx)
        elif delta == 'hour':
            valley_date = peak_date + timedelta(hours=valley_idx)
        else:
            raise ValueError(f'delta {delta} not supported')

        res.append([peak_date, x, valley_date, valley, drop])

        j += valley_idx + 1

    all_drawdowns = pd.DataFrame(res, columns=['peak_day', 'peak_price', 'valley_day', 'valley_price', 'drop_percent'])
    selected_drawdowns = all_drawdowns[all_drawdowns['drop_percent'] >= drop_min]
    return selected_drawdowns


def get_dynamic_peaks_valleys(df, e0, w):
    """
    Get all peaks and valleys defined by Epsilon Drawdown Method
    Args:
        df: the dataframe of stock price
        e0: the number of standard deviations up to which counter-movements are tolerated
        w: the lookback window length on which the standard deviation is calculated

    Returns: an array of all the peak days, an array of all the valley days

    """

    logclose = np.log(df['close'])

    peaks = []
    valleys = []
    trend = 0

    if logclose[w + 1] - logclose[w + 0] > 0:
        valleys.append(logclose.index[w + 0])
        trend = 1
    if logclose[w + 1] - logclose[w + 0] < 0:
        peaks.append(logclose.index[w + 0])
        trend = -1

    minmax = logclose[w + 1] - logclose[w + 0]
    minmax_idx = w + 1

    s = w + 0
    delta = 0
    for j in range(w + 2, df.shape[0]):
        x = logclose[j]
        p_current = x - logclose[s]

        if trend == 1:
            if p_current > minmax:
                minmax = p_current
                minmax_idx = j
            delta = minmax - p_current
        if trend == -1:
            if p_current < minmax:
                minmax = p_current
                minmax_idx = j
            delta = p_current - minmax

        epsilon = e0 * np.std(logclose[j - w:j])
        # If the reversed change delta exceeds epsilon, we consider the trend is changed, thus get a peak or valley
        if delta > epsilon:
            if trend == 1:
                peaks.append(logclose.index[minmax_idx])
            else:
                valleys.append(logclose.index[minmax_idx])
            trend *= -1
            s = minmax_idx
            minmax = x - logclose[s]
            minmax_idx = j

    return np.array(peaks), np.array(valleys)


def get_dynamic_drawdowns(df,
                          e0_start=0.1, e0_end=5, e0_step=0.1,
                          w_start=10, w_end=60, w_step=5,
                          freq_min=0.5):
    all_peaks = []
    all_valleys = []
    # Find peaks and valleys with different sets of parameters
    for e0 in np.arange(e0_start, e0_end + e0_step, e0_step):
        for w in range(w_start, w_end + w_step, w_step):
            peaks, valleys = get_dynamic_peaks_valleys(df, e0, w)
            all_peaks.append(peaks)
            all_valleys.append(valleys)

    # Calculate date frequency of peaks and valleys
    peak_freq = []
    n = len(all_peaks)
    for date in df.index:
        count = 0
        for peaks in all_peaks:
            if date in peaks:
                count += 1
        peak_freq.append(count / n)

    # Generate drawdown dataframe
    peak_freq_df = pd.DataFrame(peak_freq, index=df.index, columns=['peak_frequency'])
    my_peaks = peak_freq_df[peak_freq_df.peak_frequency > freq_min]
    end_dates, end_values, drop_percents, durations = [], [], [], []
    for i in range(my_peaks.shape[0] - 1):
        v = df[(df.index >= my_peaks.index[i]) & (df.index <= my_peaks.index[i + 1])].close.min()
        d = df[(df.index >= my_peaks.index[i]) & (df.index <= my_peaks.index[i + 1])].close.idxmin()
        end_values.append(v)
        end_dates.append(d)
        p = df.loc[my_peaks.index[i]].close
        drop_percents.append((p - v) / p)
        durations.append((d - my_peaks.index[i]).days)

    my_drawdowns = my_peaks.iloc[:-1]
    my_drawdowns['peak_price'] = df.loc[my_drawdowns.index].close
    my_drawdowns['valley_day'] = np.array(end_dates)
    my_drawdowns['valley_price'] = np.array(end_values)
    my_drawdowns['drop_percent'] = np.array(drop_percents)
    my_drawdowns['duration'] = np.array(durations)

    my_drawdowns.index.names = ['peak_day']
    my_drawdowns = my_drawdowns.reset_index()
    return my_drawdowns


def plot_updowns(df, peaks=None, valleys=None, start=None, end=None):
    if start is None:
        start = df.index[0]
    if end is None:
        end = df.index[-1]
    y = df.loc[start:end]['close']

    plt.figure(figsize=(15, 8), dpi=80)
    plt.plot(y)
    plt.xticks(rotation=45)

    if peaks is not None:
        selected_peaks = peaks[(peaks >= start) & (peaks < end)]
        plt.vlines(x=selected_peaks, ymin=min(y)-5, ymax=max(y)+5, colors='red')
    if valleys is not None:
        selected_valleys = valleys[(valleys >= start) & (valleys < end)]
        plt.vlines(x=selected_valleys, ymin=min(y)-5, ymax=max(y)+5, colors='green')
    plt.show()
