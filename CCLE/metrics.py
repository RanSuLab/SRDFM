

import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import r2_score

import pandas as pd

def pair_evaluate(y_o1, y_o2, y_pred_o1, y_pred_o2, datasize):
    real = y_o1>y_o2
    pred = y_pred_o1>y_pred_o2
    return np.sum(real == pred) * 1.0 / datasize

def point_evaluate(y_o1, y_pred_o1):

    return np.sum(y_o1-y_pred_o1)


def precision(y, f, k):
    return (1.0 * np.intersect1d(np.argsort(y)[::-1][:k], np.argsort(f)[::-1][:k]).shape[0] / k) if k > 0 else np.nan


def Precision(Y, F, k):
    ## Y: true
    ## F: predict
    n = Y.shape[0]
    precisionk = []
    for i in range(n):
        f = F[i]
        y = Y[i]
        f = f[~np.isnan(y)]
        y = y[~np.isnan(y)]
        precisionk.append(precision(y, f, min(k, y.shape[0])))
    return np.array(precisionk)

def my_precision(label, out, k):
    # print("------------------------------precision------------------------------------")
    ## label : Y : true -- only one column
    ## out: F : predict -- 3 columns (x,y,predict-value)

    Y_label = np.zeros((label.shape[0],3),dtype=float)
    Y_label[:, 0] = np.array(out)[:, 0]
    Y_label[:, 1] = np.array(out)[:, 1]
    Y_label[:, 2] = np.array(label)

    Y = np.full([491, 24], np.nan)
    for line in pd.DataFrame(Y_label).itertuples():
        # print(line[1])
        Y[int(line[1]), int(line[2])] = float(line[3])

    ## out: F
    F = np.full([491, 24], np.nan)
    for line in pd.DataFrame(out).itertuples():
        F[int(line[1]), int(line[2])] = float(line[3])

    return np.mean(Precision(Y, F, k)[~np.isnan(Precision(Y, F, k))])

def my_precision2(label, out, k):
    print("------------------------------precision------------------------------------")
    ## label : Y : true -- only one column
    ## out: F : predict -- 3 columns (x,y,predict-value)

    Y_label = np.zeros((label.shape[0],3),dtype=float)
    Y_label[:, 0] = np.array(out)[:, 0]
    Y_label[:, 1] = np.array(out)[:, 1]
    Y_label[:, 2] = np.array(label)

    Y = np.full([491, 198], np.nan)
    for line in pd.DataFrame(Y_label).itertuples():
        # print(line[1])
        Y[int(line[1]), int(line[2])] = float(line[3])

    ## out: F
    F = np.full([491, 24], np.nan)
    for line in pd.DataFrame(out).itertuples():
        F[int(line[1]), int(line[2])] = float(line[3])

    return np.mean(Precision(Y, F, k)[~np.isnan(Precision(Y, F, k))])


def evaluate_pair(y_pred_o1, y_pred_o2, y_label_o1, y_label_o2):
    datasize = len(y_pred_o1)
    # print(datasize)
    o12 = [y_pred_o1[i] - y_pred_o2[i] for i in range(len(y_pred_o1))]
    # print(o12)
    ho12 = [y_label_o1[i] - y_label_o2[i] for i in range(len(y_label_o1))]
    # print(ho12)
    symbol = [o12[i] * ho12[i] for i in range(len(o12))]
    # print(symbol)
    return np.sum(np.array(symbol)> 0)*1.0/datasize

def dcg(y, pi, k):
    return ((2 ** y[pi[:k]] - 1) / np.log(range(2, 2 + k))).sum() if k > 0 else np.nan

def ndcg(y, pi, k):
    # y：true   f: predict
    return dcg(y, pi, k) / dcg(y, np.argsort(y)[::-1], k) if k > 0 else np.nan


def NDCG(Y, F, k):
    ## Y:true
    ## F:predict

    n = Y.shape[0]
    ndcgk = []
    for i in range(n):
        f = F[i]
        y = Y[i]
        f = f[~np.isnan(y)]
        y = y[~np.isnan(y)]
        ndcgk.append(ndcg(y, np.argsort(f)[::-1], min(k, y.shape[0])))

    return np.array(ndcgk)

def my_NDCG(label, out, k):
    # print("------------------------------precision------------------------------------")

    ## label : Y : true -- only one column
    ## out: F : predict -- 3 columns (x,y,predict-value)

    Y_label = np.zeros((label.shape[0], 3), dtype=float)
    Y_label[:, 0] = np.array(out)[:, 0]
    Y_label[:, 1] = np.array(out)[:, 1]
    Y_label[:, 2] = np.array(label)

    Y = np.full([491, 24], np.nan)
    for line in pd.DataFrame(Y_label).itertuples():
        # print(line[1])
        Y[int(line[1]), int(line[2])] = float(line[3])

    ## out: F
    F = np.full([491, 24], np.nan)
    for line in pd.DataFrame(out).itertuples():
        F[int(line[1]), int(line[2])] = float(line[3])


    return np.mean(NDCG(Y, F, k)[~np.isnan(NDCG(Y, F, k))])

def rank(pool, best, which=0):
    assert which >= 0
    return list(np.argsort(pool)[::-1]).index(np.argsort(best)[::-1][which]) if pool.shape[0] > 0 else np.nan


def percentile(y, f, which):
    assert which >= 0

    return (rank(y, f, which) / float(y.shape[0])) if (y.shape[0] > 0 and y.shape[0] > which) else np.nan


def Percentile(Y, F, k):
    n = Y.shape[0]
    percentiles = []
    for i in range(n):
        y = Y[i]
        f = F[i]
        f = f[~np.isnan(y)]
        y = y[~np.isnan(y)]
        percentiles.append([percentile(y, f, which) for which in range(k)])
    return np.array(percentiles)

def my_Percentile(out, label, k):
    # print("------------------------------precision------------------------------------")
    # n_celllines, c_idx = np.unique(np.array(label)[:, 0], return_index=True)
    # n_drugs, d_idx = np.unique(np.array(label)[:, 1], return_index=True)
    ## label:Y
    Y_label = np.zeros((label.shape[0], 3), dtype=float)
    Y_label[:, 0] = np.array(out)[:, 0]
    Y_label[:, 1] = np.array(out)[:, 1]
    Y_label[:, 2] = np.array(label)

    Y = np.full([491, 24], np.nan)
    for line in pd.DataFrame(Y_label).itertuples():
        # print(line[1])
        Y[int(line[1]), int(line[2])] = float(line[3])

    ## out: F
    F = np.full([491, 24], np.nan)
    for line in pd.DataFrame(out).itertuples():
        F[int(line[1]), int(line[2])] = float(line[3])

    return np.mean(Percentile(F, Y, k)[~np.isnan(Percentile(F, Y, k))])




def rank_new(pool, best, which=0):
    assert which >= 0
    ## which = 0 : the single top durg
    return list(np.argsort(pool)[::-1]).index(np.argsort(best)[::-1][0]) if pool.shape[0] > 0 else np.nan



def Percentile_new(Y, F, k):
    #     # if which == 1 : top1 fraction
    #     # if which == 3:  top3 fraction (由top3-top1，就可以得出中间的)

    non_null_row = 0
    n = Y.shape[0]
    percentiles = []
    count = 0
    for i in range(n):
        y = Y[i]
        f = F[i]
        f = f[~np.isnan(y)]
        y = y[~np.isnan(y)]

        if y.shape[0] == 0:
            continue
        non_null_row += 1
        predict_percentile = rank_new(f, y)
        # print(" predict_percentile : %d"%predict_percentile)
        if predict_percentile < k:
            count += 1

    # print(" k : %d"%k)
    # print("count : %d" %count)
    fraction = 1.0*count / non_null_row
    # print(" fraction: %f"%fraction)
    percentiles.append(fraction)
    return np.array(percentiles)

def my_Percentile_new(label, out, k):
    print("------------------------------precision------------------------------------")

    ## label:Y:true
    ## out: F: predict

    Y_label = np.zeros((label.shape[0], 3), dtype=float)
    Y_label[:, 0] = np.array(out)[:, 0]
    Y_label[:, 1] = np.array(out)[:, 1]
    Y_label[:, 2] = np.array(label)

    Y = np.full([491, 24], np.nan)
    for line in pd.DataFrame(Y_label).itertuples():
        # print(line[1])
        Y[int(line[1]), int(line[2])] = float(line[3])

    ## out: F
    F = np.full([491, 24], np.nan)
    for line in pd.DataFrame(out).itertuples():
        F[int(line[1]), int(line[2])] = float(line[3])

    return np.mean(Percentile_new(F, Y, k)[~np.isnan(Percentile_new(F, Y, k))])

