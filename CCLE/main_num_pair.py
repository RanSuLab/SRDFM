import os

import numpy as np

import pandas as pd

import tensorflow as tf

import time


from sklearn.metrics import make_scorer

import gc

from sklearn.model_selection import KFold,StratifiedKFold
from tensorflow import set_random_seed


from DataLoader_num import FeatureDictionary, DataParser, gen_pairs, gen_pairs_local

from matplotlib import pyplot as plt

import math

import config

from metrics import pair_evaluate, my_precision, point_evaluate, evaluate_pair,my_NDCG, my_Percentile,my_Percentile_new

from Deepfm_num_pair import DeepFM

import random

def load_data():

    dfTrain = pd.read_csv(config.TRAIN_FILE)


    dfTest = pd.read_csv(config.TEST_FILE)



    cols = [c for c in dfTrain.columns if c not in ['id','id1','target','c_indices','d_indices']]

    cols = [c for c in cols if (not c in config.IGNORE_COLS)]



    X_train = dfTrain[cols].values

    print("************X_train********")

    print(X_train)

    y_cols = ['c_indices','d_indices','target']

    y_train = dfTrain[y_cols].values

    print("************y_train********")

    print(y_train)



    X_test = dfTest[cols].values

    id_cols = ['c_indices','d_indices','id']

    ids_test = dfTest[id_cols].values

    print("************ids_test********")

    print(ids_test)




    cat_features_indices = [i for i,c in enumerate(cols) if c in config.CATEGORICAL_COLS]



    return dfTrain,dfTest,X_train,y_train,X_test,ids_test,cat_features_indices



def run_base_model_dfm(dfTrain,dfTest, fig, exp, folds,dfm_params,path=None):

    fd = FeatureDictionary(drug,

                           numeric_cols=config.NUMERIC_COLS,

                           ignore_cols=config.IGNORE_COLS,

                           cate_cols=config.CATEGORICAL_COLS)


    data_parser = DataParser(feat_dict=fd)


    single_drug = dfTrain.drop_duplicates(['d_indices'], keep="first")
    print("single drug:")
    print(single_drug)

    single_drug.to_csv(path+"/single_drug.csv",index=False)

    # 与exp、fig拼接即可
    single_drug = pd.merge(single_drug, exp, on='c_indices', how="left")
    single_drug = pd.merge(single_drug, fig, on="d_indices", how="left")

    print(single_drug)

    col_ = single_drug.columns.tolist()

    un_ = []
    for each in config.NUMERIC_COLS:
        if each not in col_:
            print(each)
            un_.append(each)

    print("config.numeric: %d" % len(config.NUMERIC_COLS))
    print("un_ : %d" % len(un_))

    if len(un_) != 0:
        print("*** columns Wrong! ***")
        input()

    # print(fd.feat_dim)

    dfm_params["cate_feature_size"] = fd.feat_dim

    dfm_params["field_size"] = len(config.CATEGORICAL_COLS)

    dfm_params['numeric_feature_size'] = len(config.NUMERIC_COLS)


    _get = lambda x,l:[x[i] for i in l]

    list_k = config.list_k

    y_id_test_ = dfTest.values.tolist()


    #
    y_train_meta = np.zeros((dfTrain.shape[0], 3), dtype=float)

    y_train_meta[:, 0] = np.array(dfTrain)[:, 0]

    y_train_meta[:, 1] = np.array(dfTrain)[:, 1]

    y_test_meta = np.zeros((dfTest.shape[0], 3), dtype=float)

    y_test_meta[:, 0] = np.array(dfTest)[:, 0]

    y_test_meta[:, 1] = np.array(dfTest)[:, 1]

    #
    gini_results_cv = np.zeros(len(folds), dtype=float)

    ndcg_results_cv = np.zeros((len(folds), 6), dtype=float)

    percentile_new_results_cv = np.zeros((len(folds), 6), dtype=float)

    precision_results_cv = np.zeros((len(folds), 6), dtype=float)


    best_precision_results_cv = np.zeros((len(folds), 6), dtype=float)
    best_ndcg_results_cv = np.zeros((len(folds), 6), dtype=float)
    best_percentile_new_results_cv = np.zeros((len(folds), 6), dtype=float)



    for i, (train_idx, valid_idx) in enumerate(folds):
        

        print("fold = %d"%i)

        print(" 每次循环中，folds中的（train_idx，valid_idx）：")
        print(len(train_idx))
        print(len(valid_idx))
    #
        np.random.shuffle(train_idx)
    #
        y_id_train_list = dfTrain.values.tolist()
    #
        y_id_train_ = _get(y_id_train_list, train_idx)
    #
        y_id_valid_ = _get(y_id_train_list, valid_idx)


        y_valid_meta = np.zeros((np.array(y_id_valid_).shape[0], 4), dtype=float) # 4 =  c_id.d_id,perdicted,target

        y_valid_meta[:, 0] = np.array(y_id_valid_)[:, 0]

        y_valid_meta[:, 1] = np.array(y_id_valid_)[:, 1]


        dfm = DeepFM(**dfm_params)

        print("________________DFM over!_______________________")


        best_precision_results_cv[i],best_ndcg_results_cv[i],best_percentile_new_results_cv[i]  = dfm.fit( data_parser, exp, fig, y_id_train_, y_id_valid =y_id_valid_, y_id_test = y_id_test_,fold = i,early_stopping = True,path = path)


        print("_________ valid predict_____________")


        y_valid_meta[:, 2] = dfm.predict_train(data_parser, exp, fig, y_id_valid_)

        for j, k in enumerate(list_k):
            precision_results_cv[i, j] = my_precision(np.array(y_id_valid_)[:, 2], np.array(y_valid_meta), k)
            ndcg_results_cv[i, j] = my_NDCG(np.array(y_id_valid_)[:,2 ], np.array(y_valid_meta), k)
            percentile_new_results_cv[i, j] = my_Percentile_new(np.array(y_id_valid_)[:, 2], np.array(y_valid_meta),k)

        gini_results_cv[i] = precision_results_cv[i, 1]

        print(precision_results_cv[i])
        print(ndcg_results_cv[i])
        print(percentile_new_results_cv[i])


        ## save valid results
        y_valid_meta[:,3] = np.array(y_id_valid_)[:, 2]
        filename_valid = path + "/fold=%s_valid_result" % i
        pd.DataFrame(y_valid_meta,columns=['c_indices','d_indices','predicted','target']).to_csv(filename_valid, index=False)


        ## weights metrix

        if dfm.use_attention:
            print(" ******* drug weights : ******* ")

            dummy_y = [1] * len(single_drug)
            dummy_y = [[y_] for y_ in dummy_y[0:len(single_drug)]]
            # print(dummy_y)
            dummy_y = np.array(dummy_y)
            # print(dummy_y.shape)


            cate_Xi, cate_Xv, numeric_Xv, ids = data_parser.parse(df=single_drug)

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

            attention = attention.reshape((-1,len(config.NUMERIC_COLS)))
            # print(attention)

            attention_df = pd.DataFrame(attention,columns=config.NUMERIC_COLS)

            attention_df.to_csv(path+"/fold=%d_drug_weight.csv"%i,index=False)

        else:
            print(" no drug weights, because you do not use attetion.")



    pd.DataFrame(precision_results_cv).to_csv(path+"/precision_results_cv.csv",index=False)
    pd.DataFrame(ndcg_results_cv).to_csv(path+"/ndcg_results_cv.csv",index=False)
    pd.DataFrame(percentile_new_results_cv).to_csv(path+"/percentile_new_results_cv.csv",index=False)

    y_test_meta[:, 2] /= float(len(folds))
    
    ## anverage results

    precision_df = pd.DataFrame(best_precision_results_cv).mean()
    ndcg_df = pd.DataFrame(best_ndcg_results_cv).mean()
    percentile_new_df = pd.DataFrame(best_percentile_new_results_cv).mean()
    
    precision_df = precision_df.values.T
    ndcg_df = ndcg_df.values.T
    percentile_df = percentile_new_df.values.T

    anverage_result = np.hstack((precision_df,ndcg_df,percentile_new_df))

    print("anverage results:")
    print(anverage_result)

    pd.DataFrame(anverage_result).to_csv(path+"/best_anverage_results.csv",index = False)



    # save result

    if dfm_params["use_fm"] and dfm_params["use_deep"]:

        clf_str = "DeepFM"

    elif dfm_params["use_fm"]:

        clf_str = "FM"

    elif dfm_params["use_deep"]:

        clf_str = "DNN"

    print("%s: %.5f (%.5f)"%(clf_str, gini_results_cv.mean(), gini_results_cv.std()))

    filename = "%s_Mean%.5f_Std%.5f.csv"%(clf_str, gini_results_cv.mean(), gini_results_cv.std())

    ids_test = dfTest[['c_indices', 'd_indices', 'id']].values.tolist()

    _make_submission(ids_test, y_test_meta[:, 2], filename)


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

    plt.ylabel("Pearsonr")

    plt.title("%s"%model_name)

    plt.legend(legends)

    plt.savefig("fig/%s.png"%model_name)

    plt.close()





def number_rank(x):
    # x 向下取整
    return math.floor(x)

def normalize01(target):
    target = pd.to_numeric(target)
    # print(target)
    target_min = target.min()
    target_max = target.max()

    target_normal = (target-target_min)/(target_max-target_min)
    return target_normal


def normalize01_(target):
    target = pd.to_numeric(target)
    # print(target)
    target_min = target.min()
    target_max = target.max()

    target_normal = (target_max-target)/(target_max-target_min)
    return target_normal


def normalize(target):
    mean = target.mean()
    std = target.std()
    target_normal = (target-mean)/std
    return target_normal




###设置dfm的参数
dfm_params = {

    "use_fm":True,##dfm中的fm部分

    "use_deep":True,##dfm中的DNN部分

    "use_bn":True,

    "use_attention":False,

    "embedding_size":8,

    "dropout_fm":[1.0,1.0],

    "deep_layers":[512,256],

    "dropout_deep":[0.9,0.9,0.9,0.8,0.8],

    "deep_layer_activation":tf.nn.relu,

    "epoch":20,

    "mid_size":20000,

    "batch_size":2000,

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


print(dfm_params)


## 确定随机数
set_random_seed(config.RANDOM_SEED)
random.seed(config.RANDOM_SEED)
tf.compat.v1.set_random_seed(config.RANDOM_SEED)


## make new results dir
path = './'+time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
print(path)

isExists = os.path.exists(path)


if not isExists:
    os.mkdir(path)
    os.mkdir(path+'/ckpt0/')
    os.mkdir(path+'/ckpt1/')
    os.mkdir(path+'/ckpt2/')
    print(path + ' 创建成功')

else:
    print(path + ' 目录已存在')


# load data


dfTrain = pd.read_csv("data_fm2/ccle_actarea.csv")

dfTrain.insert(0, 'c_indices', dfTrain.pop('c_indices'))
dfTrain.insert(1, 'd_indices', dfTrain.pop('d_indices'))
dfTrain.insert(2, 'target', dfTrain.pop('target'))

#dfTrain['target'] = normalize01(dfTrain['target'])



target = dfTrain["target"].to_frame()
max_ = target.max()
min_ = target.min()
print("target_max = %f"%max_)
print("target_min = %f"%min_)

#print(dfTrain)


drug = pd.read_csv(config.DRUG_FILE)


cell = pd.read_csv(config.CELL_FILE)

'''
# #un_normalization # 1037*18989
cell = pd.read_csv("data_fm2/ccle_expr(un_normalization).csv")
## normalization
#cell = pd.read_csv("data_fm2/ccle_expr.csv")
c_match = pd.read_csv("data_fm2/cname_cindices.csv") # 2 cols
cell_ = pd.merge(cell,c_match,on="CName",how="left")
cell_.insert(0, 'c_indices', cell_.pop('c_indices'))
cell_.drop("CName",axis=1,inplace = True)
# ##
df_c1 = dfTrain["c_indices"].unique().tolist()
df_c2 = dfTest["c_indices"].unique().tolist()
df_c = list(set(df_c1+df_c2))
cell = cell_[cell_["c_indices"].isin(df_c)]
'''


target = dfTrain["target"].to_frame()
max_ = target.max()
min_ = target.min()
print("target_max = %f"%max_)
print("target_min = %f"%min_)


label = np.array(dfTrain["d_indices"].values.tolist())



# folds

print("_________________Data Load Over!______________________")

folds = list(StratifiedKFold(n_splits=config.NUM_SPLITS, shuffle=True,random_state=config.RANDOM_SEED).split(dfTrain, label))

# # save folds
for i, (train_idx, valid_idx) in enumerate(folds):

    train_name = path+"/folds_train"+str(i)+".txt"
    valid_name = path+"/folds_valid"+str(i)+".txt"
    np.savetxt(train_name, train_idx, fmt='%d', delimiter=',')
    np.savetxt(valid_name, valid_idx, fmt='%d', delimiter=',')


y_train_dfm, y_test_dfm = run_base_model_dfm(dfTrain, drug, cell, folds, dfm_params,path=path)
