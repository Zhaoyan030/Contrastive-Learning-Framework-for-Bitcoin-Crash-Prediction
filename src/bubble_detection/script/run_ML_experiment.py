import argparse
import random

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import *
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, StratifiedKFold, RepeatedStratifiedKFold
from sklearn.svm import SVC
from xgboost import XGBClassifier

from src.bubble_detection.data.utils import (
    get_modified_label, get_train_test, generate_my_features, load_model_data
)
from src.bubble_detection.model.utils import get_eval_dict, ml_output, plot_predictions

import warnings

warnings.filterwarnings('ignore')


def ml_train_eval(
        trainx, trainy, testx, testy, testy_modified,
        model_name, metric='f2', cv="stratified", verbose=0,
        rng=None, class_weights=None
):

    scoring = {
        'accuracy': make_scorer(accuracy_score),
        'f2': make_scorer(fbeta_score, beta=2),
        'recall': make_scorer(recall_score),
        "AUC": "roc_auc",
        "precision": make_scorer(precision_score)
    }
    cv_types = {
        'stratified': StratifiedKFold(n_splits=5, shuffle=True),
        'time': TimeSeriesSplit(n_splits=5),
        'repeated': RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=1)
    }

    if model_name == "RF":
        hyper = {'random_state': rng, 'class_weight': class_weights}
        model = RandomForestClassifier(**hyper)
        pipe_paras = {
            'model__n_estimators': [250],
            'model__max_depth': [8]
        }
    elif model_name == 'SVM':
        hyper = {'random_state': rng, 'probability': True}
        model = SVC(**hyper)
        pipe_paras = {
            'model__C': [100],
            'model__gamma': [0.1],
            'model__kernel': ['rbf']
        }
    elif model_name == 'GBM':
        hyper = {'random_state': rng}
        model = GradientBoostingClassifier(**hyper)
        pipe_paras = {
            # 'model__learning_rate': [0.01, 0.03, 0.1],
            'model__n_estimators': [250],
            'model__subsample': [0.8],
            'model__max_depth': [8]
        }
    elif model_name == 'XGB':
        hyper = {"class_weight": class_weights}
        model = XGBClassifier(**hyper)
        pipe_paras = {
            # 'model__learning_rate': [0.01, 0.03, 0.1],
            'model__gamma': [0.5],
            'model__subsample': [0.8],
            'model__max_depth': [8],
            'model__n_estimators': [250]
        }
    else:
        raise ValueError(f'Model type {model_name} not supported')

    # train model
    pipe = Pipeline(steps=[('scaler', RobustScaler()), ("model", model)])
    pipe_grid = GridSearchCV(
        estimator=pipe,
        param_grid=pipe_paras,
        cv=cv_types[cv],
        scoring=scoring[metric],
        verbose=verbose,
        n_jobs=-1
    )
    pipe_grid.fit(trainx, trainy)
    y_pred_prob = pipe_grid.predict_proba(testx)[:, 1]
    y_pred = pipe_grid.predict(testx)

    # Evaluate model by testy
    precision, recall, _ = precision_recall_curve(testy, y_pred_prob)
    eval_dict = get_eval_dict(testy, y_pred, y_pred_prob, precision, recall)

    # Evaluate model with margin
    precision, recall, _ = precision_recall_curve(testy_modified, y_pred_prob)
    margin_eval_dict = get_eval_dict(testy_modified, y_pred, y_pred_prob, precision, recall)

    return hyper, pipe_grid, y_pred, eval_dict, margin_eval_dict


def run_ml_experiment(
        model_name, stock, start, end, interval, train_pct, drawdowns_path, peak_frequency, drop_percent,
        win, forward, solve_imbalance=False, margin=None, myseed=None, history_path=None,
):
    df, drawdowns, y = load_model_data(
        stock=stock, start=start, end=end, interval=interval, drawdowns_path=drawdowns_path,
        peak_frequency=peak_frequency, drop_percent=drop_percent, forward=forward,
        history_path=history_path
    )
    features = generate_my_features(df, win=win)
    print(f"=== Generate {features.shape[1]} features ===")

    indices, trainx, testx, trainy, testy = get_train_test(
        features, y, model='ML', train_pct=train_pct, shuffle=False
    )

    ml_testy_modified = get_modified_label(
        testy, drawdowns, forecast_len=forward, margin=margin, margin_type='binary'
    )

    if solve_imbalance:
        w0 = 1 / (trainy.shape[0] - trainy.sum(0)) * (
                1 / (1 / (trainy.shape[0] - trainy.sum(0)) + 1 / trainy.sum(0)))
        w1 = 1 / trainy.sum(0) * (1 / (1 / (trainy.shape[0] - trainy.sum(0)) + 1 / trainy.sum(0)))
        class_weights = {0: w0, 1: w1}
        print(f"Class Weights: {class_weights}")
    else:
        # trainx, trainy = my_downsampling(trainx, trainy, win, forward)
        class_weights = None

    np.random.seed(myseed)
    rng = np.random.RandomState(myseed)
    random.seed(myseed)

    print(f"=== Training {model_name} ===")
    hyper, pipe_grid, y_pred, eval_dict, margin_eval_dict = ml_train_eval(
        trainx, trainy, testx, testy, ml_testy_modified,
        model_name, metric='f2', cv="stratified", verbose=0,
        rng=rng, class_weights=class_weights
    )

    plot = plot_predictions(df, testy, y_pred)
    ml_output(basic_info={'stock': stock, 'OUTPUT_PATH': "./output", 'label': 'dynamic', 'WIN': win, 'forward': forward, 'margin': 'n'},
              model_name=model_name, model_hyper=hyper,
              model_grid=pipe_grid, eval_dict=eval_dict, model_plot=plot)
    ml_output(basic_info={'stock': stock, 'OUTPUT_PATH': "./output", 'label': 'dynamic', 'WIN': win, 'forward': forward, 'margin': 'y'},
              model_name=model_name, model_hyper=hyper,
              model_grid=pipe_grid, eval_dict=margin_eval_dict, model_plot=None)


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
    MARGIN = {'left': 20, 'right': 0}
    for myseed in myseeds:
        run_ml_experiment(
            args.model_name, args.stock, args.start, args.end, args.interval,
            args.train_pct, args.drawdowns_path, args.peak_frequency, args.drop_percent,
            win=args.win, forward=args.forward, solve_imbalance=True,
            margin=MARGIN, myseed=myseed, history_path=args.history_path,
        )
