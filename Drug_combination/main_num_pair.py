import csv
import os
import sys
import math

import numpy as np

import pandas as pd

import tensorflow as tf

from sklearn.metrics import make_scorer

import gc

from sklearn.model_selection import KFold,StratifiedKFold

from DataLoader_num import FeatureDictionary, DataParser, gen_pairs, df_Resolve_To_numpy, Id_Index, gen_pairs_,gen_pairs_hard

from matplotlib import pyplot as plt



import config2 as config

from metrics import pair_evaluate, my_precision, point_evaluate, evaluate_pair,my_NDCG, my_Percentile,my_Percentile_new

from Deepfm_num_pair import DeepFM

import random

from sklearn.utils import shuffle



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



def load_test_data():

    dfTest = pd.read_csv(config.TEST_FILE)
    dfTest.rename(columns={'Unnamed: 0':'id1'}, inplace = True)
    dfTest.drop(columns="Unnamed: 0_y", inplace=True)
    dfTest.drop_duplicates(subset=['id'], keep='last', inplace=True)  # 重复id（key）直接覆盖掉，

    return dfTest



def run_base_model_dfm(y_id_train,exp,fig,folds,dfm_params):


    fd = FeatureDictionary(catefile = fig, numeric_cols= config.NUMERIC_COLS, ignore_cols= config.IGNORE_COLS,

                           cate_cols=config.CATEGORICAL_COLS)


    data_parser = DataParser(feat_dict=fd)



    dfm_params["cate_feature_size"] = fd.feat_dim

    dfm_params["field_size"] = len(config.CATEGORICAL_COLS)

    dfm_params['numeric_feature_size'] = len(config.NUMERIC_COLS)


    y_train_meta = np.zeros((y_id_train.shape[0], 3), dtype=float)

    y_train_meta[:, 0] = np.array(y_id_train)[:,0]

    y_train_meta[:, 1] = np.array(y_id_train)[:,1]


    _get = lambda x,l:[x[i] for i in l]


    #  ##初始化gini result
    gini_results_cv = np.zeros(len(folds), dtype=float)

    list_k = config.list_k

    ndcg_results_cv = np.zeros((len(folds),6), dtype=float)

    percentile_results_cv = np.zeros((len(folds), 6), dtype=float)

    percentile_new_results_cv = np.zeros((len(folds), 6), dtype=float)

    precision_results_cv = np.zeros((len(folds), 6), dtype=float)

    gini_results_epoch_train = np.zeros((len(folds), dfm_params['epoch']), dtype=float)

    gini_results_epoch_valid = np.zeros((len(folds), dfm_params['epoch']), dtype=float)

    best_ndcg_results_cv = np.zeros((config.NUM_SPLITS, 6), dtype=float)
    best_percentile_new_results_cv = np.zeros((config.NUM_SPLITS, 6), dtype=float)
    best_precision_results_cv = np.zeros((config.NUM_SPLITS, 6), dtype=float)



    for i, (train_idx, valid_idx) in enumerate(folds):

        y_id_train_list = y_id_train.values.tolist()

        y_id_train_ = _get(y_id_train_list,train_idx)


        y_id_valid_ = _get(y_id_train_list,valid_idx)


        dfm = DeepFM(**dfm_params)

        print("________________DFM load over!_______________________")

        best_precision_results_cv[i],best_ndcg_results_cv[i],best_percentile_new_results_cv[i] = dfm.fit( data_parser, exp, fig, y_id_train_, y_id_valid_, early_stopping=True, fold = i)


        print(" ________________DFM train over！ ________________")


        print("****** fold %d end ********"%i)


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

    pd.DataFrame(average_result).to_csv("best_anverage_results.csv", index=False)


    pd.DataFrame(best_precision_results_cv).to_csv("best_precision_results_cv.csv",index=False)
    pd.DataFrame(best_ndcg_results_cv).to_csv("best_ndcg_results_cv.csv",index=False)
    pd.DataFrame(best_percentile_new_results_cv).to_csv( "best_percentile_new_results_cv.csv",index=False)


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
    # x 向下取整
    return math.floor(x)

def normalize(target):
    # z-score
    mean = target.mean()
    std = target.std()
    target_normal = (target - mean)/std

    return target_normal

def normalize01(target):
    target = pd.to_numeric(target)
    # print(target)
    target_min = target.min()
    target_max = target.max()

    target_normal = (target - target_min) / (target_max - target_min)
    return target_normal


def dropTissue(x):
    temp = x.split("_",1)
    target = temp[0]
    return target



###设置dfm的参数
dfm_params = {

    "use_fm":True,##dfm中的fm部分

    "use_deep":True,##dfm中的DNN部分

    "use_bn": True,

    "use_attention":True,

    "embedding_size":8,

    "dropout_fm":[1.0,1.0],

    "deep_layers":[512,256], #[8196,4096,2048]

    "dropout_deep":[0.9,0.9,0.9],

    "deep_layer_activation":tf.nn.relu,

    "epoch":15,

    "batch_size":3000,

    "mid_size":3000,

    "learning_rate":0.0001,

    "optimizer":"adam",

    "batch_norm":1,

    "batch_norm_decay":0.995,

    "l2_reg":0.01,

    "verbose":True,

    "eval_metric":my_precision,

    "eval_metric_pair":evaluate_pair,

    "random_seed":config.RANDOM_SEED

}


# load data

if config.dataset == "GDSC":

    exp = pd.read_csv("data/CName_cid_GDSC.csv")
    cell = pd.read_csv("data/cgp_cell_data.csv")  # [1018 rows x 17738 columns]
    cell.rename(columns={"c_index": "c_indices"}, inplace=True)
    exp = pd.merge([exp,cell],on='c_indices',how=exp)

else:
    # CCLE
    exp = pd.read_csv("data/CName_cid_CCLE.csv")
    cell = pd.read_csv("data/cell.csv")
    exp = pd.merge([exp, cell], on='c_indices', how=exp)


cell_list = exp["CName"].unique().tolist()
#print(len(cell_list))
exp_ = exp["CName"].to_frame()
print("****** exp ok! ******")

fig = pd.read_csv("data/fingerprints.csv")
fig.drop(columns="DName_Pubc",inplace=True)
fig.drop(columns="cid",inplace=True)
fig = fig.fillna(0)
fig_c = fig.columns.tolist()
# print(fig_c)
index_ = fig.index
fig["d_indices"] = index_
# print(fig)


drug_list = fig["DName_GDSC"].values.tolist()

fig1 = fig.rename(columns = {"DName_GDSC":"drug1"},inplace=False)
fig1.rename(columns={"d_indices": "d1_indices"}, inplace=True)
fig1_ = fig1[['drug1','d1_indices']]
# print(fig1)
fig2 = fig.rename(columns={"DName_GDSC": "drug2"}, inplace=False)
fig2.rename(columns={"d_indices": "d2_indices"}, inplace=True)
fig2_ = fig2[['drug2','d2_indices']]
# print(fig2)
print("***** finger ok! ******")


### GDSC_all_pairs : 79 cell * 11996 combination * 25w rows  —— if you change the dataset, you should change the metrics.py

if config.dataset == 'GDSC':
    ## GDSC_DeepSynergy_pais:  8 cell *38 drugs * 583 combination * 4664 rows
    drug_pair = pd.read_csv("data/data_pair_DeepSynergy(GDSC).csv")
    drug_pair.drop_duplicates(subset=['drug1', 'drug2', 'CName'], keep='first', inplace=True)
elif config.dataset == 'CCLE':
    ## CCLE_DeepSynergy_pairs: 29 cell * 38 drugs * 583 combination * 16907 rows
    drug_pair = pd.read_csv("data/data_pair_DeepSynergy(CCLE).csv")
    drug_pair.drop_duplicates(subset=['drug1', 'drug2', 'CName'], keep='first', inplace=True)
else:
    print(" We only support GDSC and CCLE, if you want to use your own dataset, you should change the code.")
    exit()

drug_pair = drug_pair[( drug_pair["drug1"].isin(drug_list) & drug_pair["drug2"].isin(drug_list) )]
drug_pair = drug_pair[drug_pair["CName"].isin(cell_list)]


drug_pair = pd.merge(drug_pair,fig1_,on="drug1",how="left")
drug_pair = pd.merge(drug_pair,fig2_,on="drug2",how="left")
drug_pair.drop(columns="drug1",inplace=True)
drug_pair.drop(columns="drug2",inplace=True)


drug_pair = pd.merge(drug_pair,exp_,on="CName",how="left")

d_ = drug_pair["CName"].to_frame()
d_ = d_.drop_duplicates(keep='first').reset_index(drop = True)
d_.reset_index(inplace = True)
d_.rename(columns={"index":"c_indices"},inplace=True)

drug_pair = pd.merge(drug_pair,d_,on="CName",how="left")

drug_pair.drop(columns="CName",inplace=True)


print(" ****** drug_pair ok! ****** ")


exp = pd.merge(exp,d_,on="CName",how="right")
exp.drop(columns="CName",inplace=True)


fig1.drop(columns='drug1',inplace=True)
fig2.drop(columns="drug2",inplace=True)


## drugs_pairs
pair_df = drug_pair[["d1_indices","d2_indices"]]
pair_df = pair_df.drop_duplicates(keep='first').reset_index(drop=True)
pair_df.reset_index(inplace = True)
pair_df.rename(columns={"index":"d_indices"},inplace=True)


## fingerprint
fig_pairs = pd.merge(pair_df,fig1,on="d1_indices",how="left")
fig_pairs = pd.merge(fig_pairs,fig2,on="d2_indices",how="left")
fig_pairs.drop(columns=["d1_indices","d2_indices"],inplace=True)


drug_pair = pd.merge(drug_pair,pair_df,on=["d1_indices","d2_indices"],how="left")
drug_pair.drop(columns=["d1_indices","d2_indices"],inplace=True)


drug_pair.reset_index(inplace = True)
drug_pair.rename(columns = {"index":"id"},inplace=True)
drug_pair.insert(0, 'c_indices', drug_pair.pop('c_indices'))
drug_pair.insert(1, 'd_indices', drug_pair.pop('d_indices'))
drug_pair.insert(2, 'target', drug_pair.pop('target'))


drug_pair = shuffle(drug_pair)


drug_pair_train = drug_pair


drug_pair['target'] = normalize01(drug_pair['target'])


target = drug_pair["target"].to_frame()

max_ = target.max()
min_ = target.min()
print("target_max = %f"%max_)
print("target_min = %f"%min_)

print("_________________Data Load Over!______________________")


label = np.array(drug_pair_train["c_indices"].values.tolist())

folds = list(StratifiedKFold(n_splits=config.NUM_SPLITS, shuffle=True,random_state=config.RANDOM_SEED).split(drug_pair_train,label))


run_base_model_dfm(drug_pair_train,exp,fig_pairs, folds, dfm_params)
