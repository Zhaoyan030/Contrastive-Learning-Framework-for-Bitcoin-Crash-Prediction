import csv
from datetime import datetime
from time import time
import os

from imblearn.metrics import geometric_mean_score
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_score, recall_score, accuracy_score, auc, fbeta_score, make_scorer, \
    confusion_matrix, balanced_accuracy_score, roc_auc_score, precision_recall_curve
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, StratifiedKFold, RepeatedStratifiedKFold
import tensorflow as tf
from tslearn.metrics import dtw

from src.bubble_detection.data.augmentation import data_augmentation
from src.bubble_detection.model import tscp2 as cp2


# https://stats.stackexchange.com/questions/163054/compare-several-binary-time-series
def moving_average(true, pred, window=10, weights=None):
    df = pd.DataFrame({'true': true, 'pred': pred})
    if weights is None:
        weights = np.ones(window)
    ma_df = df.rolling(window).apply(lambda x: (x * weights).sum() / weights.sum())
    return ma_df[window:]

def ml_train(trainx, trainy, model, cv_params, metric='f2', cv="stratified", verbose=0):
    scoring = {'accuracy': make_scorer(accuracy_score),
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
    pipe = Pipeline(steps=[('scaler', RobustScaler()), ("model", model)])
    pipe_grid = GridSearchCV(estimator=pipe,
                             param_grid=cv_params,
                             cv=cv_types[cv],
                             scoring=scoring[metric],
                             verbose=verbose,
                             n_jobs=-1)
    pipe_grid.fit(trainx, trainy)
    return pipe_grid


def ml_predict_eval(testx, testy, grid, threshold=0.5):
    y_pred_prob = grid.predict_proba(testx)[:, 1]

    precision, recall, thresholds = precision_recall_curve(testy, y_pred_prob)

    if threshold == 0.5:
        y_pred = grid.predict(testx)
    else:
        fscore = (5 * precision * recall) / (4 * precision + recall)
        ix = np.argmax(fscore)
        print('Best Threshold=%f, F2=%.3f' % (thresholds[ix], fscore[ix]))
        threshold = thresholds[ix]
        y_pred = (y_pred_prob > threshold).astype(int)

    prec = precision_score(testy, y_pred)
    recl = recall_score(testy, y_pred)
    acc = accuracy_score(testy, y_pred)
    prauc = auc(recall, precision)
    rocauc = roc_auc_score(testy, y_pred_prob)
    f1 = fbeta_score(testy, y_pred, beta=1)
    f2 = fbeta_score(testy, y_pred, beta=2)
    gm = geometric_mean_score(testy, y_pred)
    blacc = balanced_accuracy_score(testy, y_pred)
    fm = (prec * recl) ** (1 / 2)
    dtw_sim = dtw(testy, y_pred)
    ma = moving_average(testy, y_pred, window=10)
    ma_dtw_sim = dtw(ma.true, ma.pred)

    eval_dict = {
        'accuracy': round(acc, 4),
        'precision': round(prec, 4),
        'recall': round(recl, 4),
        'f1': round(f1, 4),
        'f2': round(f2, 4),
        'G-mean': round(gm, 4),
        'BA': round(blacc, 4),
        'PRAUC': round(prauc, 4),
        'ROCAUC': round(rocauc, 4),
        'FM': round(fm, 4),
        'DTW': round(dtw_sim, 4),
        'MA_DTW': round(ma_dtw_sim, 4)
    }

    print("########## Grid Search Best Model ##########")
    print(grid.best_estimator_)
    print(f"Training scores: {grid.best_score_}")
    print("\n")
    print("########## Test Set Performance ##########")
    print("confusion matrix: ")
    print(confusion_matrix(testy, y_pred))
    print("precision: " + str(prec))
    print("recall:    " + str(recl))
    print("accuracy:  " + str(acc))
    print("PR AUC:    " + str(prauc))
    print("ROC AUC:   " + str(rocauc))
    print("F1:        " + str(f1))
    print("F2:        " + str(f2))
    print("G-mean:    " + str(gm))
    print("BA:        " + str(blacc))
    print("FM:        " + str(fm))
    print("DTW:       " + str(dtw_sim))
    print("MA_DTW:    " + str(ma_dtw_sim))

    return y_pred, threshold, eval_dict


def ml_output(basic_info, model_name, model_hyper, model_grid, eval_dict, model_plot=None):
    now = datetime.now().strftime("%Y%m%d%H%M")

    model_res = {
        'time': now,
        'model': model_name,
        'params': {**{'CV': 'Stratified5Fold', 'scoring': 'f2'}, **model_hyper, **model_grid.best_params_}
    }

    model_output_dict = {**basic_info, **model_res, **eval_dict}

    file_path = os.path.join(basic_info['OUTPUT_PATH'], 'ML_baseline_result.csv')

    if os.path.isfile(file_path):
        with open(file_path, 'a') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=[*model_output_dict])
            writer.writerow(model_output_dict)
            csvfile.close()
    else:
        with open(file_path, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=[*model_output_dict])
            writer.writeheader()
            writer.writerow(model_output_dict)
            csvfile.close()

    # if model_plot is not None:
    #     plot_name = model_name + "_" + basic_info['stock'] + "_" + basic_info['label'] + "_" + now + ".png"
    #     model_plot.savefig(os.path.join(basic_info['OUTPUT_PATH'], "figures", plot_name))


def plot_predictions(df, testy=None, pred_y=None, pred_prob=None):
    if testy is not None:
        close = df.loc[testy.index]['close']
    if pred_y is not None:
        pred_y = pd.Series(pred_y, index=testy.index)
        close = df.loc[pred_y.index]['close']

    fig, axs = plt.subplots(2, figsize=(20, 10), dpi=300)
    #     plt.figure(figsize=(20, 6), dpi=80)
    axs[0].plot(close)
    axs[0].set(ylabel='Price')
    axs[0].set_title('True labels in test set')
    #     plt.xticks(np.arange(0, len(close),10), rotation=45)
    if testy is not None:
        labels = testy[testy == 1]
        axs[0].vlines(x=labels.index, ymin=min(close) - 5, ymax=max(close) + 5, colors='red', alpha=0.2)

    axs[1].plot(close)
    axs[1].set(ylabel='Price')
    axs[1].set_title('Predicted labels in test set')
    if pred_y is not None:
        pred_labels = pred_y[pred_y == 1]
        axs[1].vlines(x=pred_labels.index, ymin=min(close) - 5, ymax=max(close) + 5, colors='orange', alpha=0.2)
    if pred_prob is not None:
        ax2 = axs[1].twinx()
        ax2.plot(pred_prob, color='green')
    #     plt.show()

    return fig


def dl_train_eval(model, trainx, trainy, testx, testy, testy_modified, class_weights=None):
    model, train_time, hyper = dl_train(trainx, trainy, model, class_weights)

    # all prediction
    x = np.vstack([trainx, testx])
    ally_pred_prob = model.predict(x)
    ally_pred_prob = np.squeeze(ally_pred_prob)

    # Evaluate the model with testy
    y_pred_prob = model.predict(testx)
    precision, recall, thresholds = precision_recall_curve(testy, y_pred_prob)
    y_pred = (y_pred_prob >= 0.5).astype(int)
    y_pred = np.squeeze(y_pred)
    eval_dict = get_eval_dict(testy, y_pred, y_pred_prob, precision, recall)

    # Evaluate the model with margin
    precision, recall, _ = precision_recall_curve(testy_modified, y_pred_prob)
    margin_eval_dict = get_eval_dict(testy_modified, y_pred, y_pred_prob, precision, recall)

    return hyper, ally_pred_prob, y_pred, eval_dict, margin_eval_dict, train_time


def dl_train(trainx, trainy, model, class_weights):
    hyper = {
        'lr_decay': {
            'monitor': 'loss',
            'patience': 10,
            'verbose': 0,
            'factor': 0.5,
            'min_lr': 1e-8

        },
        'early_stop': {
            'monitor': 'val_f2',
            'min_delta': 0,
            'patience': 10,
            'verbose': 1,
            'mode': 'max',
            'baseline': 0,
            'restore_best_weights': True
        }
    }

    # Define a learning rate decay method:
    lr_decay = ReduceLROnPlateau(**hyper['lr_decay'])
    # Define Early Stopping:
    early_stop = EarlyStopping(**hyper['early_stop'])

    hyper['fit'] = {
        'batch_size': 128,
        'epochs': 200,
        'validation_split': 0.2,
        'shuffle': True,
        'class_weight': class_weights,
        'callbacks': [lr_decay, early_stop]
    }

    start = time()
    model.fit(trainx, trainy, **hyper['fit'])
    train_time = round(time() - start, 2)
    return model, train_time, hyper


def get_eval_dict(testy, y_pred, y_pred_prob, precision, recall):
    prec = precision_score(testy, y_pred)
    recl = recall_score(testy, y_pred)
    acc = accuracy_score(testy, y_pred)
    prauc = auc(recall, precision)
    rocauc = roc_auc_score(testy, y_pred_prob)
    f1 = fbeta_score(testy, y_pred, beta=1)
    f2 = fbeta_score(testy, y_pred, beta=2)
    gm = geometric_mean_score(testy, y_pred)
    blacc = balanced_accuracy_score(testy, y_pred)
    fm = (prec * recl) ** (1 / 2)
    dtw_sim = dtw(testy, y_pred)
    ma = moving_average(testy, y_pred)
    ma_dtw_sim = dtw(ma.true, ma.pred)
    eval_dict = {
        'accuracy': round(acc, 4),
        'precision': round(prec, 4),
        'recall': round(recl, 4),
        'f1': round(f1, 4),
        'f2': round(f2, 4),
        'G-mean': round(gm, 4),
        'BA': round(blacc, 4),
        'PRAUC': round(prauc, 4),
        'ROCAUC': round(rocauc, 4),
        'FM': round(fm, 4),
        "DTW": round(dtw_sim, 4),
        "MA_DTW": round(ma_dtw_sim, 4)
    }
    return eval_dict


def dl_output(basic_info, model_name, model, eval_dict, train_time, model_plot=None) -> object:
    now = datetime.now().strftime("%Y%m%d%H%M")

    model_res = {
        'time': now,
        'model': model_name,
        'total_paras': model.count_params(),
        'train_time': train_time
    }

    model_output_dict = {**basic_info, **model_res, **eval_dict}

    file_path = os.path.join(basic_info['OUTPUT_PATH'], 'DL_baseline_result.csv')

    if os.path.isfile(file_path):
        with open(file_path, 'a') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=[*model_output_dict])
            writer.writerow(model_output_dict)
            csvfile.close()
    else:
        with open(file_path, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=[*model_output_dict])
            writer.writeheader()
            writer.writerow(model_output_dict)
            csvfile.close()

    # if model_plot is not None:
    #     plot_name = model_name + "_" + basic_info['stock'] + "_" + basic_info['label'] + "_" + now + ".png"
    #     model_plot.savefig(os.path.join(basic_info['OUTPUT_PATH'], "figures", plot_name))


def cl_get_representation(
        cl_trainx, cl_testx, WIN,
        aug1_method, aug2_method, aug1_method_param, aug2_method_param, comb,
        code_size, nb_filters, kernel_size, nb_stacks, dilations, padding, dropout_rate, return_sequences,
        BATCH_SIZE, LR, EPOCHS, TEMP, similarity, LOSS, BETA, TAU
):
    cl_trainx_aug = data_augmentation(cl_trainx,
                                      aug1_method,
                                      aug2_method,
                                      aug1_method_param,
                                      aug2_method_param,
                                      comb)
    np.random.shuffle(cl_trainx_aug)

    # transform trainx into the format for TSCP2
    cp2_trainx = tf.data.Dataset.from_tensor_slices(cl_trainx_aug)
    cp2_trainx = (cp2_trainx.batch(BATCH_SIZE, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE))

    # set the similarity measurement
    #     similarity = ls._cosine_simililarity_dim2

    # train the model
    prep_model = cp2.get_TCN_encoder((WIN, 1),
                                     int(WIN / 2),
                                     code_size,
                                     nb_filters,
                                     kernel_size,
                                     nb_stacks,
                                     dilations,
                                     padding,
                                     dropout_rate,
                                     return_sequences)
    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=LR)
    epoch_wise_loss, epoch_wise_sim, epoch_wise_neg, prep_model = cp2.train_prep(prep_model,
                                                                                 cp2_trainx,
                                                                                 optimizer,
                                                                                 WIN,
                                                                                 temperature=TEMP,
                                                                                 epochs=EPOCHS,
                                                                                 sfn=similarity,
                                                                                 lfn=LOSS,
                                                                                 beta=BETA,
                                                                                 tau=TAU)

    cl_trainx_rep, _ = prep_model(cl_trainx)
    cl_testx_rep, _ = prep_model(cl_testx)

    cl_trainx_rep = np.array(cl_trainx_rep)
    cl_testx_rep = np.array(cl_testx_rep)
    return cl_trainx_rep, cl_testx_rep


def cl_get_encoder(model, cl_trainx, WIN, subsample_pct=1,
                   aug1_method='time_warp', aug2_method="mag_warp", aug1_method_param={}, aug2_method_param={}, comb=False,
                   code_size=16, nb_filters=64, kernel_size=4, nb_stacks=2, dilations=None, padding='causal', dropout_rate=0,return_sequences=True,
                   channels=1, layers=[64,64],
                   BATCH_SIZE=32, LR=1e-4, LR_DECAY=0.1, EPOCHS=200, TEMP=0.1, similarity='cosine', LOSS='ntxent', BETA=1, TAU=0.1,
                   save_encoder=False):
    cl_trainx_aug = data_augmentation(cl_trainx,
                                      aug1_method,
                                      aug2_method,
                                      aug1_method_param,
                                      aug2_method_param,
                                      comb)
    np.random.shuffle(cl_trainx_aug)
    cl_trainx_aug = cl_trainx_aug[:int(cl_trainx_aug.shape[0] * subsample_pct)]

    # transform trainx into the format for TSCP2
    cp2_trainx = tf.data.Dataset.from_tensor_slices(cl_trainx_aug)
    cp2_trainx = (cp2_trainx.batch(BATCH_SIZE, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE))

    # set the similarity measurement
    #     similarity = ls._cosine_simililarity_dim2

    # train the model
    if model == "TCN":
        prep_model = cp2.get_TCN_encoder((WIN, 1),
                                         # int(WIN / 2),
                                         WIN,
                                         code_size,
                                         nb_filters,
                                         kernel_size,
                                         nb_stacks,
                                         dilations,
                                         padding,
                                         dropout_rate,
                                         return_sequences)
    elif model == "LSTM":
        prep_model = cp2.get_LSTM_encoder(cp2_trainx, WIN=WIN, channels=channels, code_size=code_size, layers=layers)
    else:
        raise ValueError("Model not recognized")

    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=LR, decay=LR_DECAY)
    epoch_wise_loss, epoch_wise_sim, epoch_wise_neg, prep_model = cp2.train_prep(
        prep_model, cp2_trainx, optimizer, WIN, temperature=TEMP,
        epochs=EPOCHS, sfn=similarity, lfn=LOSS, beta=BETA, tau=TAU
    )
    model_name = "CL_encoder" + "_" + model
    if save_encoder:
        prep_model.save(os.path.join("./output/models", model_name))
    return epoch_wise_loss, epoch_wise_sim, epoch_wise_neg, prep_model, model_name