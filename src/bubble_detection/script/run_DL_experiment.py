import argparse
import random

import numpy as np
import pandas as pd
import tensorflow as tf

from src.bubble_detection.data.utils import get_train_test, get_modified_label, load_model_data
from src.bubble_detection.model.classifiers import get_LSTM_classifier, get_TCN_classifier
from src.bubble_detection.model.utils import plot_predictions, dl_output, dl_train_eval

import warnings
warnings.filterwarnings('ignore')


def run_dl_exprmt(
        model_name, stock, start, end, interval, train_pct, drawdowns_path, peak_frequency, drop_percent,
        win, forward, history_path=None, margin=None, myseed=None, save_prediction=False, solve_imbalance=False,
):

    df, drawdowns, y = load_model_data(
        stock=stock, start=start, end=end, interval=interval, drawdowns_path=drawdowns_path,
        peak_frequency=peak_frequency, drop_percent=drop_percent, forward=forward,
        history_path=history_path
    )

    indices, trainx, testx, trainy, testy = get_train_test(
        df, y, model='DL', train_pct=train_pct, win=win, shuffle=False,
        # cols=['open', 'high', 'low', 'close', 'volume', 'diff', 'return', 'vol']
        cols=['close', 'return'],
    )
    testy_modified = get_modified_label(
        testy, drawdowns, forecast_len=forward, margin=margin, margin_type='binary'
    )

    if solve_imbalance:
        w0 = 1 / (trainy.shape[0] - trainy.sum(0)) * (
                1 / (1 / (trainy.shape[0] - trainy.sum(0)) + 1 / trainy.sum(0)))
        w1 = 1 / trainy.sum(0) * (1 / (1 / (trainy.shape[0] - trainy.sum(0)) + 1 / trainy.sum(0)))
        class_weights = {0: w0, 1: w1}
    else:
        class_weights = {0: 0.5, 1: 0.5}
        # class_weights = {0: 0.1, 1: 0.9}
    print(f"Class Weights: {class_weights}")

    random.seed(myseed)
    np.random.seed(myseed)
    tf.random.set_seed(myseed)

    if model_name == 'LSTM':
        model = get_LSTM_classifier(trainx, WIN=win, channels=trainx.shape[2], code_size=12, layers=[64, 64])
    elif model_name == "TCN":
        model = get_TCN_classifier(WIN=win, channels=trainx.shape[2], code_size=12, nb_filters=64, dilations=[1,2,4,8])
    else:
        raise ValueError("Model name must be either 'LSTM' or 'TCN'")

    hyper, ally_pred_prob, y_pred, eval_dict, margin_eval_dict, train_time = dl_train_eval(
        model=model, trainx=trainx, trainy=trainy, testx=testx, testy=testy, testy_modified=testy_modified,
        class_weights=class_weights
    )

    test_plot = plot_predictions(df, testy, np.squeeze(y_pred))

    dl_output(
        basic_info={'stock': stock, 'OUTPUT_PATH': "./output", 'label': 'dynamic', 'WIN': win, 'forward': forward,
                    'margin': 'n'},
        model_name=model_name, model=model, eval_dict=eval_dict, train_time=train_time,
        model_plot=test_plot)

    dl_output(
        basic_info={'stock': stock, 'OUTPUT_PATH': "./output", 'label': 'dynamic', 'WIN': win, 'forward': forward,
                    'margin': 'y'},
        model_name=model_name, model=model, eval_dict=margin_eval_dict, train_time=0, model_plot=None)

    # Manual save code for full prediction
    if save_prediction:
        csv_name = f'output/full_pred_{model_name}.csv'
        try:
            full_pred = pd.read_csv(csv_name, parse_dates=['date']).set_index('date')
            full_pred[f'seed_{myseed}'] = pd.Series(ally_pred_prob, index=indices)
        except:
            print("Warning: csv file doesn't exist, create a new one.")
            full_pred = pd.DataFrame({'date': indices, f'seed_{myseed}': ally_pred_prob})
        full_pred.to_csv(csv_name, index=False)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Run CL experiments")
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--stock",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--start",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--end",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--interval",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--train_pct",
        type=float,
        required=True,
    )

    parser.add_argument(
        "--drawdowns_path",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--history_path",
        type=str,
        required=False,
        default=None,
    )

    parser.add_argument(
        "--peak_frequency",
        type=float,
        required=True,
    )

    parser.add_argument(
        "--drop_percent",
        type=float,
        required=True,
    )

    parser.add_argument(
        "--win",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--forward",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--iteration",
        action='store_true',
    )

    args = parser.parse_args()

    if args.iteration:
        myseeds = np.arange(10)
    else:
        myseeds = [0]
    MARGIN = {'left': 30, 'right': 0}
    for myseed in myseeds:
        run_dl_exprmt(
            model_name=args.model_name, stock=args.stock, start=args.start, end=args.end, interval=args.interval,
            train_pct=args.train_pct, drawdowns_path=args.drawdowns_path, history_path=args.history_path,
            peak_frequency=args.peak_frequency, drop_percent=args.drop_percent,
            win=args.win, forward=args.forward,
            margin=MARGIN, myseed=myseed, save_prediction=True, solve_imbalance=False
        )