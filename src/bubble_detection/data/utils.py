from math import floor

import numpy as np
import pandas as pd

from src.bubble_detection.data.dataset import StockDataset

STOCK_TICKER = {
    'Bitcoin': 'BTC-USD',
    'SP500': '^GSPC',
}


def load_model_data(
        stock, drawdowns_path, drop_percent, peak_frequency, forward,
        history_path=None, start=None, end=None, interval=None,
):
    ticker = STOCK_TICKER[stock]
    drawdowns = pd.read_csv(drawdowns_path, parse_dates=['peak_day', 'valley_day'])
    if 'peak_frequency' in drawdowns.columns:
        drawdowns = drawdowns[
            (drawdowns['peak_frequency'] >= peak_frequency) & (drawdowns['drop_percent'] >= drop_percent)
            ]
    else:
        drawdowns = drawdowns[drawdowns['drop_percent'] >= drop_percent]

    # Load data from Yahoo if local data path not provided
    if history_path:
        if 'hourly' in history_path:
            df = pd.read_csv(history_path, parse_dates=['datetime'], index_col=['datetime'])
        else:
            df = pd.read_csv(history_path, parse_dates=['date'], index_col=['date'])
    else:
        dataset = StockDataset(ticker)
        df = dataset.get_hist(start=start, end=end, interval=interval)
    y = get_drawdown_label(df, drawdowns, forecast_len=forward)
    return df, drawdowns, y


def sliding_window(df: pd.DataFrame, window_len: int, step_size: int = 1, index='end'):
    """
    Generate sliding windows and their date indices.

    Args:
        df: a hist dataframe generated from dataset.py
        window_len: length of sliding windows
        step_size: step size when sliding forward
        index: which index is used for each windows -- 'end' means using the end date of the window, 'start' means
                  using the start date of the window

    Returns: a list of window indices, and a 3-d array with shape (# windows, length of window, # features)

    """
    indices, windows = [], []
    if index == 'end':
        for i in range(0, df.shape[0] - window_len + 1, step_size):
            indices.append(df.index[i + window_len - 1])
            windows.append(df.iloc[i:i + window_len].values)
    if index == 'start':
        for i in range(0, df.shape[0] - window_len + 1, step_size):
            indices.append(df.index[i])
            windows.append(df.iloc[i:i + window_len].values)
    return indices, np.stack(windows)


def get_drawdown_label(df: pd.DataFrame, drawdowns: pd.DataFrame, forecast_len: int = 5):
    """
    Generate y () for each date in df, mark period that is forecast_len days before drawdown happens (peak) to 1.

    Args:
        df: the hist df of StockDataset class
        drawdowns: a drawdown table
        forecast_len: number of days to forecast ahead

    Returns: pd.Series

    """

    labels = pd.Series(0, index=df.index)
    for date in drawdowns['peak_day']:
        if date in labels.index:
            iloc = labels.index.get_loc(date)
            if iloc < forecast_len:
                labels[:iloc] = 1
            else:
                labels[iloc - forecast_len + 1:iloc + 1] = 1
    return labels


def generate_my_features(df: pd.DataFrame, win: int = 40):
    # df.index = df.index.strftime('%Y-%m-%d')
    df['return_vol'] = df['return'].rolling(10).std()

    # moving average
    ma_win = np.arange(5, win, 5)
    moving_features = {}
    for ma in ma_win:
        close_column_name = f"close_MA_for_{ma}_wins"
        moving_features[close_column_name] = df['close'].rolling(ma).mean()

        return_column_name = f"return_MA_for_{ma}_wins"
        moving_features[return_column_name] = df['return'].rolling(ma).mean()

        vol_column_name = f"vol_MA_for_{ma}_wins"
        moving_features[vol_column_name] = df['return_vol'].rolling(ma).mean()
    moving_features = pd.DataFrame.from_dict(moving_features)

    return moving_features


def get_train_test(x, y, model='ML', train_pct=0.8, win=30, shuffle=False, cols=None, step_size=1):
    """
    Split all windows into training set and test set.

    Args:
        x: a data frame of features
        y: a np.Series containing labels
        model: a string ("ML"， "DL", or "CL") indicating the train and test sets are generated for what kind of model
        train_pct: a float < 1, the percentage of training set
        win: window length
        shuffle: if shuffle training set
        cols: column names
        step_size: step size for sliding windows

    Returns:
        indices:
        trainx: shuffled training set
        testx: test set
        trainy: training label matched to trainx
        testy: test label matched testx

    """

    if cols is None:
        cols = ['close']
    if model == "ML":
        x = x.dropna()
        indices = x.index.intersection(y.index)
        x = x.loc[indices]
        y = y.loc[indices]
        train_size = int(floor(train_pct * x.shape[0]))
        idx = np.arange(train_size)
        if shuffle:
            np.random.shuffle(idx)

        trainx = x.iloc[idx]
        testx = x.iloc[train_size:]
        trainy = y[idx]
        testy = y[train_size:]

    elif model == "DL":
        indices, windows = sliding_window(x[cols], window_len=win, step_size=step_size)
        train_size = int(floor(train_pct * windows.shape[0]))
        idx = np.arange(train_size)

        trainx = windows[idx]
        testx = windows[train_size:]
        trainy = y[indices][idx]
        testy = y[indices][train_size:]

    elif model == "CL":
        indices, windows = sliding_window(x[cols], window_len=win, step_size=step_size)
        train_size = int(floor(train_pct * windows.shape[0]))
        idx = np.arange(train_size)

        trainx = np.squeeze(windows[idx])
        testx = np.squeeze(windows[train_size:])
        trainy = y[indices][idx]
        testy = y[indices][train_size:]

    else:
        raise ValueError("Invalid model, only ['ML', 'DL', 'CL'] are supported")

    print("##### Size of data sets for {} models #####".format(model))
    print("The shape of training set: {}".format(trainx.shape))
    print("The shape of test set: {}".format(testx.shape))
    print("The shape of training label: {}".format(trainy.shape))
    print("The percentage of class 1 in training label: {}%".format(round(trainy.sum() / trainy.shape[0] * 100, 4)))
    print("The shape of test label: {}".format(testy.shape))
    print("The percentage of class 1 in test label: {}%".format(round(testy.sum() / testy.shape[0] * 100, 4)))

    return indices, trainx, testx, trainy, testy


def my_downsampling(trainx, trainy, WIN, forward):
    subsample_idx = []
    i = 0
    while i < len(trainy) - 1:
        if trainy[i] == 1:
            subsample_idx.append(i)
            if trainy[i + 1] == 0:
                for j in range(int(WIN / 2), int(WIN / 2) + forward + 1):
                    if i + j < len(trainy) and trainy[i + j] == 0:
                        subsample_idx.append(i + j)
                    else:
                        break
                i = i + min(j, int(WIN / 2) + forward)
            else:
                i += 1
        else:
            i += 1

    if type(trainx) is pd.core.frame.DataFrame:
        trainx_sub = trainx.iloc[subsample_idx]
    else:
        trainx_sub = trainx[subsample_idx]
    trainy_sub = trainy[subsample_idx]
    print("The percentage of class 1 in sub-training label: {}%".format(
        round(trainy_sub.sum() / trainy_sub.shape[0] * 100, 4)))

    return trainx_sub, trainy_sub


def get_modified_label(y, drawdowns, forecast_len=5, margin={'left': 30, 'right': 5}, margin_type='binary'):
    """
    generate y () for each date in df, according to the drawdowns
    :param df: the hist df of StockDataset class
    :param drawdowns: a drawdown table
    :param forecast_len: number of days to forecast ahead
    :return: a pd.Series
    """
    labels = pd.Series(0.0, index=y.index)
    left = margin['left']
    right = margin['right']
    for date in drawdowns['peak_day']:
        if date in labels.index:
            iloc = labels.index.get_loc(date)

            if margin_type == 'binary':
                if iloc < left:
                    labels[:iloc] = 1
                else:
                    labels[iloc - left + 1: iloc + right + 1] = 1

            if margin_type == "weighted":
                if iloc < forecast_len:
                    labels[:iloc] = 1
                else:
                    labels[iloc - forecast_len + 1:iloc + 1] = 1
                    for j in range(forecast_len + 1, left + 1):
                        labels[iloc - j + 1] = 1 / (j - forecast_len)
                    for k in range(right):
                        labels[iloc + k + 1] = 1 / (k + 2)
    return labels


def my_accuracy(y_true, y_pred, weights):
    """
    calculate my accuracy
    :param y_true:
    :param y_pred:
    :param weights: the output of get_modified_label(testy, drawdowns, forecast_len=5, margin_len=[30, 5],
                                                     margin_type='weighted')
    :return:
    """
    diff = abs(y_true - y_pred) * (1 - weights)
    acc = 1 - diff.sum() / y_true.shape[0]
    return acc


def scale_windows(windows, model="DL", scaler='CompareToFirst'):
    scaled_windows = []
    for window in windows:
        if scaler == 'CompareToFirst':
            scaled_window = [(p / window[0] - 1) for p in window]
        elif scaler == "Normalize":
            scaled_window = (window - window.mean()) / window.std()
        elif scaler == "MinMax":
            scaled_window = (window - window.min()) / (window.max()- window.min())

        if model == "DL":
            scaled_window = np.reshape(scaled_window, (len(scaled_window), 1))
        scaled_windows.append(scaled_window)
    return np.array(scaled_windows)