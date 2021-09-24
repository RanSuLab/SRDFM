import csv
import os
import sys
import math

import time

import numpy as np

import pandas as pd

import tensorflow as tf

from sklearn.metrics import make_scorer

import gc

from sklearn.model_selection import KFold,StratifiedKFold

from DataLoader_num import FeatureDictionary, DataParser, gen_pairs, df_Resolve_To_numpy, Id_Index, gen_pairs_

from matplotlib import pyplot as plt



import config2 as config

from metrics import pair_evaluate, my_precision, point_evaluate, evaluate_pair,my_NDCG, my_Percentile,my_Percentile_new

from Deepfm_num_pair import DeepFM

import random

def load_train_data():


    dfTrain = pd.read_csv(config.TRAIN_FILE)
    dfTrain.rename(columns={'Unnamed: 0':'id1'}, inplace = True)
    dfTrain.drop(columns="Unnamed: 0_y",inplace=True)
    print(dfTrain.columns)

    cols = [c for c in dfTrain.columns if c not in ['id','id1','target','c_indices','d_indices']]

    cols = [c for c in cols if (not c in config.IGNORE_COLS)]

    X_train = dfTrain[cols].values

    print("************X_train********")

    print(X_train)

    y_cols = ['c_indices','d_indices','target']

    y_train = dfTrain[y_cols].values

    print("************y_train********")

    print(y_train)

    return dfTrain,X_train,y_train



def run_base_model_dfm(y_id_train,exp,fig,folds,dfm_params):

    fd = FeatureDictionary(catefile = fig, numeric_cols= config.NUMERIC_COLS, ignore_cols= config.IGNORE_COLS,

                           cate_cols=config.CATEGORICAL_COLS)

    data_parser = DataParser(feat_dict=fd)

    single_drug = y_id_train.drop_duplicates(['d_indices'], keep="first")
    print("single drug:")
    print(single_drug.shape)

    single_drug.to_csv(saved_path+"single_drug_df.csv",index=False)
    print(" ##### single drug saved to csv file ##### ")


    # concat with exp、fig
    single_drug = pd.merge(single_drug, exp, on='c_indices', how="left")
    single_drug = pd.merge(single_drug, fig, on="d_indices", how="left")
    # print(single_drug) ## c_indices  d_indices    target

    col_ = single_drug.columns.tolist()

    un_ = []
    for each in config.NUMERIC_COLS:
        if each not in col_:
            print(each)
            un_.append(each)


    print("config.numeric: %d" % len(config.NUMERIC_COLS))
    print("un_ : %d" % len(un_))

    if len(un_) != 0:
        print("*** main_num_pair.py #215 : columns Wrong! ***")
        exit()

    ## attention矩阵
    attention_array = np.zeros((len(single_drug), len(config.NUMERIC_COLS)), dtype=float)

    dummy_y = [1] * len(single_drug)
    dummy_y = [[y_] for y_ in dummy_y[0:len(single_drug)]]
    # print(dummy_y)
    dummy_y = np.array(dummy_y)
    # print(dummy_y.shape)


    cate_Xi, cate_Xv, numeric_Xv, ids = data_parser.parse(df=single_drug)

    del single_drug
    gc.collect()


    dfm_params["cate_feature_size"] = fd.feat_dim

    dfm_params["field_size"] = len(config.CATEGORICAL_COLS)

    dfm_params['numeric_feature_size'] = len(config.NUMERIC_COLS)

    y_train_meta = np.zeros((y_id_train.shape[0], 3), dtype=float)

    y_train_meta[:, 0] = np.array(y_id_train)[:,0]

    y_train_meta[:, 1] = np.array(y_id_train)[:,1]


    _get = lambda x,l:[x[i] for i in l]


    #
    gini_results_cv = np.zeros(len(folds), dtype=float)

    list_k = config.list_k

    ndcg_results_cv = np.zeros((len(folds),6), dtype=float)

    percentile_new_results_cv = np.zeros((len(folds), 6), dtype=float)

    precision_results_cv = np.zeros((len(folds), 6), dtype=float)

    gini_results_epoch_train = np.zeros((len(folds), dfm_params['epoch']), dtype=float)

    gini_results_epoch_valid = np.zeros((len(folds), dfm_params['epoch']), dtype=float)


    y_id_train_list = y_id_train.values.tolist()

    best_ndcg_results_cv = np.zeros((config.NUM_SPLITS, 6), dtype=float)
    best_percentile_new_results_cv = np.zeros((config.NUM_SPLITS, 6), dtype=float)
    best_precision_results_cv = np.zeros((config.NUM_SPLITS, 6), dtype=float)


    for i, (train_idx, valid_idx) in enumerate(folds):
        print("My fold = %d"%i)

        y_id_train_ = _get(y_id_train_list,train_idx)
        y_id_valid_ = _get(y_id_train_list,valid_idx)

        y_valid_meta = np.zeros((np.array(y_id_valid_).shape[0], 4), dtype=float)  ## 4 =  c_id.d_id,perdicted,target

        y_valid_meta[:, 0] = np.array(y_id_valid_)[:, 0]

        y_valid_meta[:, 1] = np.array(y_id_valid_)[:, 1]


        dfm = DeepFM(**dfm_params)

        print("________________DFM load over!_______________________")

        best_precision_results_cv[i],best_ndcg_results_cv[i],best_percentile_new_results_cv[i] = dfm.fit( data_parser, exp, fig, y_id_train_, y_id_valid=y_id_valid_, early_stopping=True, fold = i,path = saved_path)


        print("__________ valid predict_____________")


        y_valid_meta[:, 2] = dfm.predict_train(data_parser, exp, fig, y_id_valid_)

        for j, k in enumerate(list_k):
            precision_results_cv[i, j] = my_precision(np.array(y_id_valid_)[:, 2], np.array(y_valid_meta), k)
            ndcg_results_cv[i, j] = my_NDCG(np.array(y_id_valid_)[:,2], np.array(y_valid_meta), k)
            percentile_new_results_cv[i, j] = my_Percentile_new(np.array(y_id_valid_)[:, 2], np.array(y_valid_meta),k)


        # print("precision_results_cv[%d]" % i)
        print(precision_results_cv[i])
        # print("ndcg_results_cv[%d]" % i)
        print(ndcg_results_cv[i])
        # print("percentile_new_results_cv[%d]" % i)
        print(percentile_new_results_cv[i])


        ## save valid results
        y_valid_meta[:, 3] = np.array(y_id_valid_)[:, 2]
        filename_valid = saved_path+"fold=%s_valid_result" % i
        pd.DataFrame(y_valid_meta, columns=['c_indices', 'd_indices', 'predicted', 'target']).to_csv(filename_valid,
                                                                                                  index=False)


        # response

        if dfm.use_attention:
            print(" ******* drug weights : ******* ")

            feed_dict = {dfm.feat_index_o1: cate_Xi,

                         dfm.feat_value_o1: cate_Xv,

                         dfm.numeric_value_o1: numeric_Xv,

                         dfm.label_o1: dummy_y,

                         dfm.feat_index_o2: cate_Xi,

                         dfm.feat_value_o2: cate_Xv,

                         dfm.numeric_value_o2: numeric_Xv,

                         dfm.label_o2: dummy_y,

                         dfm.dropout_keep_fm: dfm.dropout_fm,

                         dfm.dropout_keep_deep: dfm.dropout_dep,

                         dfm.train_phase: False}

            attention_list = dfm.sess.run(
                [dfm.attention_o1], feed_dict=feed_dict)

            # print(attention_list)

            attention = attention_list[0]

            attention = np.array(attention)

            print("attention.shape:")
            print(attention.shape)

            attention = attention.reshape((-1, len(config.NUMERIC_COLS)))
            # print(attention)

            attention_array += attention

            print("attention_array.shape:")
            print(attention_array.shape)

        else:
            print(" no drug weights, because you do not use attetion.")


        print("****** fold %d end ********" % i)


    pd.DataFrame(precision_results_cv).to_csv(saved_path+"precision_results_cv.csv",index=False)
    pd.DataFrame(ndcg_results_cv).to_csv(saved_path+"ndcg_results_cv.csv",index=False)
    pd.DataFrame(percentile_new_results_cv).to_csv(saved_path+"percentile_new_results_cv.csv", index=False)

    
    ### weights 均值
    attention_array /= float(len(folds))
    attention_df = pd.DataFrame(attention_array, columns=config.NUMERIC_COLS)
    attention_df.to_csv(saved_path+"average_weights.csv", index=False)


    ## anverage result
    percision_df = pd.DataFrame(best_precision_results_cv).mean()
    ndcg_df = pd.DataFrame(best_ndcg_results_cv).mean()
    percentile_new_df = pd.DataFrame(best_percentile_new_results_cv).mean()

    percision_df = percision_df.values.T
    ndcg_df = ndcg_df.values.T
    percentile_new_df = percentile_new_df.values.T

    average_result = np.hstack((percision_df, ndcg_df, percentile_new_df))

    print("average results:")
    print(average_result)

    pd.DataFrame(average_result).to_csv(path + "/best_anverage_results.csv", index=False)


    # save result

    if dfm_params["use_fm"] and dfm_params["use_deep"]:

        clf_str = "DeepFM"

    elif dfm_params["use_fm"]:

        clf_str = "FM"

    elif dfm_params["use_deep"]:

        clf_str = "DNN"

    print("%s: %.5f (%.5f)"%(clf_str, gini_results_cv.mean(), gini_results_cv.std()))

    return 0



def _make_submission(ids, y_pred, filename="submission.csv"):

    pd.DataFrame({"id": ids, "target": y_pred.flatten()}).to_csv(

        os.path.join(config.SUB_DIR, filename), index=False, float_format="%.5f")





def _plot_fig(train_results, valid_results, model_name):

    colors = ["red", "blue", "green"]

    xs = np.arange(1, train_results.shape[1]+1)

    plt.figure()

    legends = []

    for i in range(train_results.shape[0]):

        plt.plot(xs, train_results[i], color=colors[i], linestyle="solid", marker="o")

        plt.plot(xs, valid_results[i], color=colors[i], linestyle="dashed", marker="o")

        legends.append("train-%d"%(i+1))

        legends.append("valid-%d"%(i+1))

    plt.xlabel("Epoch")

    plt.ylabel("Normalized Gini")

    plt.title("%s"%model_name)

    plt.legend(legends)

    plt.savefig("fig/%s.png"%model_name)

    plt.close()


def number_rank(x):
    return math.floor(x)


def normalize01(target):
    target = pd.to_numeric(target)
    target_min = target.min()
    target_max = target.max()

    target_normal = (target-target_min)/(target_max-target_min)
    return target_normal

def normalize01_(target):
    target = pd.to_numeric(target)
    target_min = target.min()
    target_max = target.max()

    target_normal = (target_max-target)/(target_max-target_min)
    return target_normal


dfm_params = {

    "use_fm":True,

    "use_deep":True,

    "use_bn": True,

    "use_attention":True,

    "embedding_size":8,

    "dropout_fm":[1.0,1.0],

    "deep_layers":[1024,512], #[8196,4096,2048]

    "dropout_deep":[0.9,0.9,0.9,0.9],

    "deep_layer_activation":tf.nn.relu,

    "epoch":30,

    "batch_size":3000,

    "mid_size":6000,

    "learning_rate":0.0001,

    "optimizer":"adam",

    "batch_norm":1,

    "batch_norm_decay":0.995,

    "l2_reg":0,

    "verbose":True,

    "eval_metric":my_precision,

    "eval_metric_pair":evaluate_pair,

    "random_seed":config.RANDOM_SEED

}



method = ""

if dfm_params["use_attention"]:
    if dfm_params["use_fm"]:
        method = "ADFM/"
    else:
        method = "ADNN/"
elif dfm_params["use_fm"]:
    method = "DFM/"
else:
    method = "DNN/"

## make new results dir
path = method+time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
isExists = os.path.exists(path)
if not isExists:
    os.makedirs(path)
    print(path + ' mkdir success!')
else:
    print(path + ' already exist!')

saved_path = path +'/'

# load data
dfTrain = pd.read_csv(config.TRAIN_FILE)
dfTrain.insert(0, 'c_indices', dfTrain.pop('c_indices'))
dfTrain.insert(1, 'd_indices', dfTrain.pop('d_indices'))
dfTrain.insert(2, 'target', dfTrain.pop('target'))
dfTrain['target'] = normalize01(dfTrain['target'])


fig = pd.read_csv(config.DRUG_FILE)

exp = pd.read_csv(config.CELL_FILE)


print("_________________Data Load Over!______________________")


label = np.array(dfTrain['d_indices'].values.tolist())
folds = list(StratifiedKFold(n_splits=config.NUM_SPLITS, shuffle=True,random_state=config.RANDOM_SEED).split(dfTrain,label))

# save folds
for i, (train_idx, valid_idx) in enumerate(folds):
    train_name = path+"/folds_train"+str(i)+".txt"
    valid_name = path+"/folds_valid"+str(i)+".txt"
    np.savetxt(train_name, train_idx, fmt='%d', delimiter=',')
    np.savetxt(valid_name, valid_idx, fmt='%d', delimiter=',')

run_base_model_dfm(dfTrain,exp,fig, folds, dfm_params)
