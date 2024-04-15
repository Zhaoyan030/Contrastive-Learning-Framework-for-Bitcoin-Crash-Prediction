import argparse
import random
from time import time

import numpy as np
import pandas as pd
import tensorflow as tf

from src.bubble_detection.data.utils import get_train_test, get_modified_label, load_model_data
from src.bubble_detection.model.classifiers import get_linear_classifier
from src.bubble_detection.model.utils import cl_get_encoder, plot_predictions, dl_output, dl_train_eval

import warnings
warnings.filterwarnings('ignore')


def run_cl_exprmt(
        stock, start, end, interval, train_pct, drawdowns_path, peak_frequency, drop_percent,
        aug_method1, aug_method2, aug_method1_param, aug_method2_param, comb,
        myseed, encoder_type='TCN', WIN=30, FORWARD=7, BATCH=16, CODE_SIZE=16, NB_FILTERS=64,
        KERNEL_SIZE=4, NB_STACK=2, DILATIONS=None, LOSS="ntxent", SUBSAMPLE=1,
        TEMP=0.1, TAU=0.1, BETA=1, margin={'left': 20, 'right': 0}, solve_imbalance=False,
        SAVE_PRED=False, history_path=None,
):

    df, drawdowns, y = load_model_data(
        stock=stock, start=start, end=end, interval=interval, drawdowns_path=drawdowns_path,
        peak_frequency=peak_frequency, drop_percent=drop_percent, forward=FORWARD,
        history_path=history_path
    )

    _, cp2_trainx, _, _, _ = get_train_test(
        df, y, model='CL', train_pct=train_pct, win=WIN, shuffle=False, step_size=5
    )
    # cp2_trainx = scale_windows(cp2_trainx.squeeze(), model="CL", scaler="MinMax")

    # CL encoder training params
    params = {
        'df_col': 'close',
        'window': WIN,
        'forward_step': FORWARD,
        'train_encoder': {
            'BATCH_SIZE': BATCH,
            'EPOCHS': 200,
            'LR': 1e-2,
            'LR_DECAY': 0.5,
            'similarity': 'cosine',
            'LOSS': LOSS,
            'TEMP': TEMP,
            'TAU': TAU,
            'BETA': BETA,
            'subsample_pct': SUBSAMPLE
        },
        'data_augmentation': {
            'aug1_method': aug_method1,
            'aug1_method_param': aug_method1_param,
            'aug2_method': aug_method2,
            'aug2_method_param': aug_method2_param,
            'comb': comb

        },
        'TCN': {
            'code_size': CODE_SIZE,
            'nb_filters': NB_FILTERS,
            'kernel_size': KERNEL_SIZE,
            'nb_stacks': NB_STACK,
            'dilations': DILATIONS,
            'padding': 'causal',
            'return_sequences': False,
            'dropout_rate': 0
        },
        'LSTM': {
            'channels': 1,
            'code_size': 12,
            'layers': [64, 64]
        }
    }

    random.seed(myseed)
    np.random.seed(myseed)
    tf.random.set_seed(myseed)

    # Train CL Encoder
    encoder_start = time()
    epoch_wise_loss, epoch_wise_sim, epoch_wise_neg, prep_model, model_name = cl_get_encoder(
        encoder_type,
        cp2_trainx,
        WIN,
        **params['data_augmentation'],
        **params[encoder_type],
        **params['train_encoder'],
        save_encoder=False
    )
    encoder_time = round(time() - encoder_start, 3)

    # Train down-streaming classifier
    indices, trainx, testx, trainy, testy = get_train_test(
        df, y, model='CL', train_pct=train_pct, win=WIN, shuffle=False, step_size=1
    )
    testy_modified = get_modified_label(
        testy, drawdowns, forecast_len=FORWARD, margin=margin, margin_type='binary'
    )
    # trainx = scale_windows(trainx.squeeze(), model="CL", scaler="MinMax")
    # testx = scale_windows(testx.squeeze(), model="CL", scaler="MinMax")
    trainx_rep, testx_rep = np.array(prep_model(trainx)), np.array(prep_model(testx))
    clf = get_linear_classifier()

    if solve_imbalance:
        w0 = 1 / (trainy.shape[0] - trainy.sum(0)) * (
                1 / (1 / (trainy.shape[0] - trainy.sum(0)) + 1 / trainy.sum(0)))
        w1 = 1 / trainy.sum(0) * (1 / (1 / (trainy.shape[0] - trainy.sum(0)) + 1 / trainy.sum(0)))
        class_weights = {0: w0, 1: w1}
    else:
        class_weights = {0: 0.5, 1: 0.5}
        # class_weights = {0: 0.1, 1: 0.9}
    print(f"Class Weights: {class_weights}")

    hyper, ally_pred_prob, y_pred, eval_dict, margin_eval_dict, train_time = dl_train_eval(
        model=clf, trainx=trainx_rep, trainy=trainy, testx=testx_rep, testy=testy, testy_modified=testy_modified,
        class_weights=class_weights
    )
    test_plot = plot_predictions(df, testy, np.squeeze(y_pred))

    # Save experiment results
    dl_output(
        basic_info={'stock': stock, 'OUTPUT_PATH': "./output", 'label': 'dynamic', 'WIN': WIN, 'forward': FORWARD,
                    'margin': 'n'},
        model_name=model_name, model=clf, eval_dict=eval_dict, train_time=encoder_time + train_time,
        model_plot=test_plot
    )

    dl_output(
        basic_info={'stock': stock, 'OUTPUT_PATH': "./output", 'label': 'dynamic', 'WIN': WIN, 'forward': FORWARD,
                    'margin': 'y'},
        model_name=model_name, model=clf, eval_dict=margin_eval_dict, train_time=0, model_plot=None
    )

    if SAVE_PRED:
        csv_name = 'output/full_pred_CL.csv'
        try:
            full_pred = pd.read_csv(csv_name, parse_dates=['date']).set_index('date')
            full_pred[f'seed_{myseed}_win{WIN}'] = pd.Series(ally_pred_prob, index=indices)
        except:
            print("Warning: csv file doesn't exist, create a new one.")
            full_pred = pd.DataFrame({'date': indices, f'seed_{myseed}_win{WIN}': ally_pred_prob})
        full_pred.to_csv(csv_name, index=True)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Run CL experiments")
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
        default=None
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
        "--encoder_type",
        type=str,
        required=True,
        choices=['TCN', 'LSTM']
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
        run_cl_exprmt(
            stock=args.stock, start=args.start, end=args.end, interval=args.interval,
            train_pct=args.train_pct, drawdowns_path=args.drawdowns_path,
            peak_frequency=args.peak_frequency, drop_percent=args.drop_percent,
            aug_method1='identity', aug_method1_param={},
            aug_method2='time_warp', aug_method2_param={'knot': 8, 'sigma': 0.6}, comb=False,
            myseed=myseed, encoder_type=args.encoder_type, history_path=args.history_path,
            WIN=args.win, FORWARD=args.forward, BATCH=16, CODE_SIZE=12, NB_FILTERS=64,
            KERNEL_SIZE=4, NB_STACK=2, DILATIONS=[1, 2, 4, 8], LOSS="ntxent", SUBSAMPLE=1,
            TEMP=0.1, TAU=0.1, BETA=1, margin=MARGIN, solve_imbalance=True, SAVE_PRED=True
        )
