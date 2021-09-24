import gc
from random import random

import numpy as np

import tensorflow as tf
from sklearn.model_selection import KFold
from tensorflow import set_random_seed

from DataLoader_num import gen_pairs_,gen_pairs_local,gen_pairs_by_cell_line

from time import time

from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.metrics import roc_auc_score

from tensorflow.contrib.layers import xavier_initializer,variance_scaling_initializer

# from cosine_loss import cosine_loss
from metrics import my_precision,my_NDCG,my_Percentile,my_Percentile_new

from metrics_gdsc import pair_evaluate, my_precision, point_evaluate, evaluate_pair,my_NDCG, my_Percentile,my_Percentile_new

import pandas as pd

import config

import os


tf.set_random_seed(config.RANDOM_SEED)

class DeepFM(BaseEstimator, TransformerMixin):


    def __init__(self, cate_feature_size, field_size, numeric_feature_size, fold,

                 embedding_size=8, dropout_fm=[1.0, 1.0],

                 deep_layers=[32, 32], dropout_deep=[0.5, 0.5, 0.5],

                 deep_layer_activation=tf.nn.relu,

                 epoch=1, batch_size=256, mid_size = 5000,

                 learning_rate=0.001, optimizer="adam",

                 batch_norm=0, batch_norm_decay=0.995,

                 verbose=False, random_seed=2016,

                 use_fm=True, use_deep=True, use_bn = False, use_attention = False,

                 loss_type="cosine", eval_metric=roc_auc_score,eval_metric_pair=roc_auc_score,

                 l2_reg=0.0, greater_is_better=True,

                 k_recommend=[3]):

        assert (use_fm or use_deep)

        assert loss_type in ["logloss", "mse", "cosine"], "loss_type can be either 'logloss' for classification task or 'mse' for regression task or 'cosine' for recommendation"

        self.cate_feature_size = cate_feature_size

        self.numeric_feature_size = numeric_feature_size

        self.field_size = field_size

        self.embedding_size = embedding_size

        self.total_size = self.field_size * self.embedding_size + self.numeric_feature_size

        self.dropout_fm = dropout_fm

        self.deep_layers = deep_layers

        self.dropout_dep = dropout_deep

        self.deep_layers_activation = deep_layer_activation

        self.use_fm = use_fm

        self.use_deep = use_deep

        self.use_bn = use_bn

        self.use_attention = use_attention

        self.l2_reg = l2_reg



        self.epoch = epoch

        self.batch_size = batch_size

        self.mid_size = mid_size

        self.learning_rate = learning_rate

        self.optimizer_type = optimizer

        self.fold = fold



        self.batch_norm = batch_norm

        self.batch_norm_decay = batch_norm_decay



        self.verbose = verbose

        self.random_seed = random_seed

        self.loss_type = loss_type

        self.eval_metric = eval_metric

        self.eval_metric_pair = eval_metric_pair

        self.greater_is_better = greater_is_better

        self.train_result,self.valid_result = [],[]

        self.k_recommend = k_recommend


        self._init_graph()



    def _init_graph(self):

        self.graph = tf.Graph()

        with self.graph.as_default():

            tf.set_random_seed(self.random_seed)



            self.feat_index_o1 = tf.placeholder(tf.int32,

                                             shape=[None,None],

                                             name='feat_index_o1')

            self.feat_value_o1 = tf.placeholder(tf.float32,

                                           shape=[None,None],

                                           name='feat_value_o1')


            self.numeric_value_o1 = tf.placeholder(tf.float32,[None,None],name='num_value_o1')

            self.label_o1 = tf.placeholder(tf.float32,shape=[None, 1],name='label_o1')

            self.feat_index_o2 = tf.placeholder(tf.int32,

                                             shape=[None,None],

                                             name='feat_index_o2')

            self.feat_value_o2 = tf.placeholder(tf.float32,

                                           shape=[None,None],

                                           name='feat_value_o2')


            self.numeric_value_o2 = tf.placeholder(tf.float32,[None,None],name='num_value_o2')

            self.label_o2 = tf.placeholder(tf.float32,shape=[None, 1],name='label_o2')

            self.dropout_keep_fm = tf.placeholder(tf.float32,shape=[None],name='dropout_keep_fm')

            self.dropout_keep_deep = tf.placeholder(tf.float32,shape=[None],name='dropout_deep_deep')

            self.train_phase = tf.placeholder(tf.bool,name='train_phase')


            self.weights = self._initialize_weights()



            # model


            self.embeddings_o1 = tf.nn.embedding_lookup(self.weights['feature_embeddings'],self.feat_index_o1) # N * 881 * 8

            feat_value_o1 = tf.reshape(self.feat_value_o1,shape=[-1,self.field_size,1]) # N * 881 *1

            self.embeddings_o1 = tf.multiply(self.embeddings_o1,feat_value_o1)  # N * 881 * 8


            # attention:  gene weight

            if self.use_attention:

                self.attention_o1_ = tf.add(tf.matmul( tf.reshape(self.embeddings_o1,shape=[-1,self.field_size * self.embedding_size]), self.weights["attention_w"]),self.weights["attention_b"])

                self.attention_o1 = tf.sigmoid(self.attention_o1_)  # 0-1

                #print(self.attention_o1)

                self.numeric_o1 = tf.multiply(self.numeric_value_o1, self.attention_o1)

                print(self.numeric_o1)


                self.x0_o1 = tf.concat([self.numeric_o1,

                                     tf.reshape(self.embeddings_o1,shape=[-1,self.field_size * self.embedding_size])]

                                    ,axis=1)

            else:
                self.x0_o1 = tf.concat([self.numeric_value_o1,

                                        tf.reshape(self.embeddings_o1,
                                                   shape=[-1, self.field_size * self.embedding_size])]

                                       , axis=1)

            # ### batch_normalization
            if self.use_bn:
                self.x0_o1 = tf.reshape(self.x0_o1, shape=[-1, self.total_size])

                print(self.x0_o1.shape)  # N * 26036

                self.x0_o1 = tf.layers.batch_normalization(self.x0_o1, training=self.train_phase)



            self.embeddings_o2 = tf.nn.embedding_lookup(self.weights['feature_embeddings'],self.feat_index_o2) # N * 881 * 8

            feat_value_o2 = tf.reshape(self.feat_value_o2,shape=[-1,self.field_size,1]) # N * 881 *1

            self.embeddings_o2 = tf.multiply(self.embeddings_o2,feat_value_o2)  # N * 881 * 8

            ## attention

            if self.use_attention:

                self.attention_o2_ = tf.add(tf.matmul(tf.reshape(self.embeddings_o2,shape=[-1,self.field_size * self.embedding_size]), self.weights["attention_w"]),self.weights["attention_b"])

                self.attention_o2 = tf.sigmoid(self.attention_o2_)  # 0-1

                self.numeric_o2 = tf.multiply(self.numeric_value_o2, self.attention_o2)


                self.x0_o2 = tf.concat([self.numeric_o2,

                                     tf.reshape(self.embeddings_o2,shape=[-1,self.field_size * self.embedding_size])]

                                    ,axis=1)
            else:
                self.x0_o2 = tf.concat([self.numeric_value_o2,

                                        tf.reshape(self.embeddings_o2,
                                                   shape=[-1, self.field_size * self.embedding_size])]

                                       , axis=1)

            # ### batch_normalization
            if self.use_bn:
                self.x0_o2 = tf.reshape(self.x0_o2, shape=[-1, self.total_size])

                print(self.x0_o2.shape)  # N * 26036

                self.x0_o2 = tf.layers.batch_normalization(self.x0_o2, training=self.train_phase)



            # first order term (W*X)

            weights_feature_bias = tf.reshape(self.weights['feature_bias'],shape=[-1,self.total_size,1]) # 1*26036*1

            features_o1 = tf.reshape(self.x0_o1,shape=[-1,self.total_size,1]) # N*26036*1

            self.y_first_order_o1 = tf.reduce_sum(tf.multiply(weights_feature_bias,features_o1),2) # N*26036

            self.y_first_order_o1 = tf.nn.dropout(self.y_first_order_o1,self.dropout_keep_fm[0])




            weights_feature_bias = tf.reshape(self.weights['feature_bias'],shape=[-1,self.total_size,1]) # 1*26036*1

            features_o2 = tf.reshape(self.x0_o2,shape=[-1,self.total_size,1]) # N*26036*1

            self.y_first_order_o2 = tf.reduce_sum(tf.multiply(weights_feature_bias,features_o2),2) # N*26036

            self.y_first_order_o2 = tf.nn.dropout(self.y_first_order_o2,self.dropout_keep_fm[0])






            # second order term

            # matmul-square-part

            self.matmul_features_vec_o1 = tf.matmul(self.x0_o1,self.weights['feature_vector']) # N * 8

            self.matmul_features_vec_square_o1 = tf.pow(self.matmul_features_vec_o1,2) # N * 8




            self.matmul_features_vec_o2 = tf.matmul(self.x0_o2,self.weights['feature_vector']) # N * 8

            self.matmul_features_vec_square_o2 = tf.pow(self.matmul_features_vec_o2,2) # N * 8



            # squre-sum-part

            self.squared_matmul_features_vec_o1 = tf.matmul(tf.pow(self.x0_o1,2),tf.pow(self.weights['feature_vector'],2))  # N * 8


            self.squared_matmul_features_vec_o2 = tf.matmul(tf.pow(self.x0_o2,2),tf.pow(self.weights['feature_vector'],2))  # N * 8


            #second order

            self.y_second_order_o1 = 0.5 * tf.subtract(self.matmul_features_vec_square_o1,self.squared_matmul_features_vec_o1)  # N * 8

            self.y_second_order_o1 = tf.nn.dropout(self.y_second_order_o1,self.dropout_keep_fm[1])




            self.y_second_order_o2 = 0.5 * tf.subtract(self.matmul_features_vec_square_o2,self.squared_matmul_features_vec_o2)  # N * 8

            self.y_second_order_o2 = tf.nn.dropout(self.y_second_order_o2,self.dropout_keep_fm[1])




            # Deep component

            self.y_deep_o1 = tf.nn.dropout(self.x0_o1, self.dropout_keep_deep[0])



            for i in range(0,len(self.deep_layers)):
            ###################y_deep = w*X +b
                self.y_deep_o1 = tf.add(tf.matmul(self.y_deep_o1,self.weights["layer_%d" %i]), self.weights["bias_%d"%i])

                # if self.use_bn:
                #     self.y_deep_o1 = tf.layers.batch_normalization(self.y_deep_o1, training=self.train_phase)

                self.y_deep_o1 = self.deep_layers_activation(self.y_deep_o1)

                self.y_deep_o1 = tf.nn.dropout(self.y_deep_o1,self.dropout_keep_deep[i+1])




            self.y_deep_o2 = tf.nn.dropout(self.x0_o2, self.dropout_keep_deep[0])

            for i in range(0, len(self.deep_layers)):
                ###################y_deep = w*X +b
                self.y_deep_o2 = tf.add(tf.matmul(self.y_deep_o2, self.weights["layer_%d" % i]), self.weights["bias_%d" % i])

                # if self.use_bn:
                #     self.y_deep_o2 = tf.layers.batch_normalization(self.y_deep_o2, training=self.train_phase)

                self.y_deep_o2 = self.deep_layers_activation(self.y_deep_o2)

                self.y_deep_o2 = tf.nn.dropout(self.y_deep_o2, self.dropout_keep_deep[i + 1])



            #----DeepFM---------


            if self.use_fm and self.use_deep:

                concat_input_o1 = tf.concat([self.y_first_order_o1, self.y_second_order_o1, self.y_deep_o1], axis=1)

                concat_input_o2 = tf.concat([self.y_first_order_o2, self.y_second_order_o2, self.y_deep_o2], axis=1)

            elif self.use_fm:

                concat_input_o1 = tf.concat([self.y_first_order_o1, self.y_second_order_o1], axis=1)

                concat_input_o2 = tf.concat([self.y_first_order_o2, self.y_second_order_o2], axis=1)

            elif self.use_deep:

                concat_input_o1 = self.y_deep_o1

                concat_input_o2 = self.y_deep_o2


#####################out = w*X + b
            self.out_o1 = tf.add(tf.matmul(concat_input_o1,self.weights['concat_projection']),self.weights['concat_bias'])

            self.out_o2 = tf.add(tf.matmul(concat_input_o2, self.weights['concat_projection']),
                                 self.weights['concat_bias'])



            # loss

            # if self.loss_type == "logloss":
            #
            #     self.out = tf.nn.sigmoid(self.out)
            #
            #     self.label_new = tf.reshape(self.label[:, 2], shape=[-1, 1])
            #
            #     self.loss = tf.losses.log_loss(self.label_new, self.out)
            #
            # elif self.loss_type == "mse":
            #
            #     self.label_new = self.label[:, 2]
            #
            #     #print(self.label_new.shape)
            #
            #     self.label_new = tf.reshape(self.label_new, shape=[-1, 1])
            #
            #     self.loss = tf.nn.l2_loss(tf.subtract(self.label_new, self.out))
            #
            # elif self.loss_type == "cosine":
            #
            #     self.out = tf.nn.sigmoid(self.out)
            #
            #     self.label_new = tf.reshape(self.label[:, 2], shape=[-1, 1])
            #
            #     self.loss = cosine_loss(self.label_new, self.out)
            # self.label_o1 = tf.reshape(self.label_o1, shape=[-1, 1])
            # self.label_o2 = tf.reshape(self.label_o2, shape=[-1, 1])
            self.o12 = tf.subtract(self.label_o1,self.label_o2)
            self.h_o12 = tf.subtract(self.out_o1, self.out_o2)

            # pred = 1 / (1 + tf.exp(-h_o12))
            # lable_p = 1 / (1 + tf.exp(-o12))
            #
            # cross_entropy = -lable_p * tf.log(pred) - (1 - lable_p) * tf.log(1 - pred) #N*1
            # reduce_sum = tf.reduce_sum(cross_entropy, 1) #N


            self.a = tf.square(tf.subtract(self.o12,self.h_o12))
            self.loss = tf.reduce_mean(self.a)#1


            # l2 regularization on weights

            # if self.l2_reg > 0:
            #
            #     self.loss += tf.contrib.layers.l2_regularizer(
            #
            #         self.l2_reg)(self.weights["concat_projection"])
            #
            #     if self.use_deep:
            #
            #         for i in range(len(self.deep_layers)):
            #
            #             self.loss += tf.contrib.layers.l2_regularizer(
            #
            #                 self.l2_reg)(self.weights["layer_%d" % i])



            if self.optimizer_type == "adam":

                print("adam*********")

                self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.9, beta2=0.999,

                                                        epsilon=1e-8)

            elif self.optimizer_type == "adagrad":

                print("adagrad*********")

                self.optimizer = tf.train.AdagradOptimizer(learning_rate=self.learning_rate,

                                                           initial_accumulator_value=1e-8)

            elif self.optimizer_type == "gd":
                # self.gd = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).compute_gradients(self.loss)

                self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)

            elif self.optimizer_type == "momentum":

                self.optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=0.95)



            ### update parameters

            # BN
            if self.use_bn:
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

                with tf.control_dependencies(update_ops):
                    self.optimizer = self.optimizer.minimize(self.loss)
            else:
                self.optimizer = self.optimizer.minimize(self.loss)




            #init



            # init = tf.global_variables_initializer()

            self.sess = tf.Session()
            # self.sess.run(init)
            model_file = tf.train.latest_checkpoint('CCLE_model/model6/ckpt%d/'%self.fold)
            self.saver = tf.train.Saver()
            self.saver.restore(self.sess, model_file)


            # number of params

            total_parameters = 0

            for variable in self.weights.values():

                shape = variable.get_shape()

                variable_parameters = 1

                for dim in shape:

                    variable_parameters *= dim.value

                total_parameters += variable_parameters

            if self.verbose > 0:

                print("#params: %d" % total_parameters)


    def predict_train(self, data_parser, exp, fig, y_id):

        """
        ## predict() 函数不考虑 pair，直接取batch预测

        :param Xi: list of list of feature indices of each sample in the dataset

        :param Xv: list of list of feature values of each sample in the dataset

        :param ids_cd: [c_id, d_id]

        :return: predicted probability of each sample

        """

        dummy_y = [1] * len(y_id)  # 行数

        print("Un-Pair Predict start!")

        batch_index = 0

        Xi_batch_o1, Xv_batch_o1, Xv2_batch_o1, y_batch_o1 = self.get_batch_feature_train(data_parser, exp, fig, y_id,
                                                                                          dummy_y, self.batch_size,
                                                                                          batch_index)

        y_pred_o1 = None

        while len(Xi_batch_o1) > 0:

            num_batch = len(Xi_batch_o1)

            feed_dict = {self.feat_index_o1: Xi_batch_o1,

                         self.feat_value_o1: Xv_batch_o1,

                         self.label_o1: y_batch_o1,

                         self.numeric_value_o1: Xv2_batch_o1,

                         self.feat_index_o2: Xi_batch_o1,

                         self.feat_value_o2: Xv_batch_o1,

                         self.label_o2: y_batch_o1,

                         self.numeric_value_o2: Xv2_batch_o1,

                         self.dropout_keep_fm: self.dropout_fm,

                         self.dropout_keep_deep: [1.0] * len(self.dropout_dep),

                         self.train_phase: True}

            batch_out_o1, projection, optimizer = self.sess.run(
                [self.out_o1, self.weights['concat_projection'], self.optimizer], feed_dict=feed_dict)

            # print(optimizer)

            # print(" saved_model_prediction.py #610, concat_projection:")
            # print(projection)

            if batch_index == 0:

                y_pred_o1 = np.reshape(batch_out_o1, (num_batch,))



            else:

                y_pred_o1 = np.concatenate((y_pred_o1, np.reshape(batch_out_o1, (num_batch,))))

            # print("~~~~~~~~~~~~~~~~~y_pred:~~~~~~~~~~~~~~~~~~~~~")
            #
            # print(y_pred)

            batch_index += 1

            # Xi_batch_o1, Xv_batch_o1, Xv2_batch_o1, y_batch_o1 = self.get_batch_feature(Xi, Xv, Xv2,dummy_y, self.batch_size, batch_index)
            Xi_batch_o1, Xv_batch_o1, Xv2_batch_o1, y_batch_o1 = self.get_batch_feature_train(data_parser, exp, fig,
                                                                                              y_id, dummy_y,
                                                                                              self.batch_size,
                                                                                              batch_index)

        return y_pred_o1

    def _initialize_weights(self):

        weights = dict()


        #embeddings

        weights['feature_embeddings'] = tf.Variable(

            tf.random_normal([self.cate_feature_size,self.embedding_size],0.0,0.01),

            name='feature_embeddings')

        weights['feature_bias'] = tf.Variable(tf.random_normal([self.total_size,1],0.0,1.0),name='feature_bias')

        weights['feature_vector'] = tf.Variable(

            tf.random_normal([self.total_size,self.embedding_size],0.0,0.01),

            name='feature_vector')

        # ###  he_initial
        # weights['feature_embeddings'] = tf.get_variable(name='feature_embeddings', shape = [self.cate_feature_size,self.embedding_size], dtype=tf.float32, initializer=variance_scaling_initializer())
        #
        # weights['feature_bias'] = tf.get_variable(name='feature_bias', shape = [self.total_size,1], dtype=tf.float32, initializer=variance_scaling_initializer())
        #
        # weights['feature_vector']  = tf.get_variable(name='feature_vector', shape = [self.total_size,self.embedding_size], dtype=tf.float32, initializer=variance_scaling_initializer())



        #deep layers

        num_layer = len(self.deep_layers)

        input_size = self.total_size

        glorot = np.sqrt(2.0/(input_size + self.deep_layers[0]))

        np.random.seed(self.random_seed)



        weights['layer_0'] = tf.Variable(

            np.random.normal(loc=0,scale=glorot,size=(input_size,self.deep_layers[0])),dtype=np.float32

        )

        weights['bias_0'] = tf.Variable(

            np.random.normal(loc=0,scale=glorot,size=(1,self.deep_layers[0])),dtype=np.float32

        )


        for i in range(1,num_layer):

            glorot = np.sqrt(2.0 / (self.deep_layers[i - 1] + self.deep_layers[i]))

            weights["layer_%d" % i] = tf.Variable(

                np.random.normal(loc=0, scale=glorot, size=(self.deep_layers[i - 1], self.deep_layers[i])),

                dtype=np.float32)  # layers[i-1] * layers[i]

            # ### he_initial
            # weights["layer_%d" % i] = tf.get_variable(name = 'layer_%d'%i,shape = [self.deep_layers[i - 1], self.deep_layers[i]],dtype=np.float32,initializer=variance_scaling_initializer())

            weights["bias_%d" % i] = tf.Variable(

                np.random.normal(loc=0, scale=glorot, size=(1, self.deep_layers[i])),

                dtype=np.float32)  # 1 * layer[i]


        # final concat projection layer



        if self.use_fm and self.use_deep:

            input_size = self.total_size + self.embedding_size + self.deep_layers[-1]
            # print("Weight矩阵初始化：input——size：")
            # print(input_size)

        elif self.use_fm:

            input_size = self.total_size + self.embedding_size

        elif self.use_deep:

            input_size = self.deep_layers[-1]



        glorot = np.sqrt(2.0/(input_size + 1))

        weights['concat_projection'] = tf.Variable(np.random.normal(loc=0,scale=glorot,size=(input_size,1)),dtype=np.float32)

        weights['concat_bias'] = tf.Variable(tf.constant(0.01),dtype=np.float32)

        # ### he_initial
        # weights['concat_projection'] = tf.get_variable(name='concat_projection',shape=[input_size,1],
        #                                             dtype=tf.float32, initializer=variance_scaling_initializer())


        # attention

        if self.use_attention:

            # glorot = np.sqrt(2.0 / (self.field_size * self.embedding_size + self.numeric_feature_size))
            #
            # weights["attention_w"] = tf.Variable(np.random.normal(loc=0, scale=glorot, size=(self.field_size* self.embedding_size, self.numeric_feature_size)), dtype=np.float32)
            #
            # weights["attention_b"] = tf.Variable(np.random.normal(loc=0, scale=glorot, size=(1, self.numeric_feature_size)), dtype=np.float32)

            ### he_initial
            #weights["attention_c"] = tf.get_variable(name='attention_c', shape=[1, self.numeric_feature_size],
                                                     #dtype=tf.float32, initializer=variance_scaling_initializer())

            weights['attention_w'] = tf.get_variable(name='attention_w', shape=[self.field_size*self.embedding_size ,self.numeric_feature_size], dtype=tf.float32, initializer=variance_scaling_initializer())

            weights['attention_b'] = tf.get_variable(name='attention_b', shape=[1, self.numeric_feature_size], dtype=tf.float32, initializer=variance_scaling_initializer())


        return weights

    def get_batch_feature_train(self,data_parser, exp,fig,y_id, y, batch_size,index):

        start = index * batch_size

        end = (index + 1) * batch_size

        end = end if end < len(y_id) else len(y_id)

        y_id_batch = y_id[start:end]

        y_id_df = pd.DataFrame(y_id_batch, columns=config.cols)

        batchTrain = pd.merge(y_id_df, exp, on='c_indices', how="left")
        batchTrain = pd.merge(batchTrain, fig, on="d_indices", how="left")
        # print("After merge, batchTrain:")
        # print(batchTrain)

        cate_Xi_valid_, cate_Xv_valid_, numeric_Xv_valid_,ids = data_parser.parse(df=batchTrain)

        del batchTrain
        gc.collect()

        del y_id_df
        gc.collect()



        return cate_Xi_valid_,cate_Xv_valid_,numeric_Xv_valid_,[[y_] for y_ in y[start:end]]


class FeatureDictionary(object):

    def __init__(self, catefile=None,

                 numeric_cols=[],

                 ignore_cols=[],

                 cate_cols=[]):



        self.catefile = catefile

        #self.testfile = testfile

        # self.testfile = testfile

        self.cate_cols = cate_cols

        self.numeric_cols = numeric_cols

        self.ignore_cols = ignore_cols

        self.gen_feat_dict()



    def gen_feat_dict(self):

        df = self.catefile[self.cate_cols]

        self.feat_dict = {}

        self.feat_len = {}

        tc = 0

        for col in df.columns:

            if col in self.ignore_cols or col in self.numeric_cols:

                continue

            else:
#############无重复的、由大到小的元组
                us = df[col].unique()
# 生成index：将US和系数一一配对，这就是zip的含义
                self.feat_dict[col] = dict(zip(us, range(tc, len(us) + tc)))

                tc += len(us)

        self.feat_dim = tc


class DataParser(object):

    def __init__(self, feat_dict):

        self.feat_dict = feat_dict





    def parse(self, infile=None, df=None, has_label=False):

        assert not ((infile is None) and (df is None)), "infile or df at least one is set"

        assert not ((infile is not None) and (df is not None)), "only one can be set"

        if infile is None:

            dfi = df.copy()

        else:

            dfi = pd.read_csv(infile)

        if has_label:

            y_cols = ['c_indices', 'd_indices', 'target']

            y = dfi[y_cols].values.tolist()

            # y_cols = ['target']
            #
            # y = dfi[y_cols].values.tolist()

            id_cols = ['c_indices', 'd_indices', 'id']

            ids = dfi[id_cols].values.tolist()

            dfi.drop(["id", "target", "c_indices", "d_indices"], axis=1, inplace=True)

        else:

            id_cols = ['c_indices', 'd_indices', 'id']

            ids = dfi[id_cols].values.tolist()

            dfi.drop(["id", "target", "c_indices", "d_indices"], axis=1, inplace=True)

        # dfi for feature index

        # dfv for feature value which can be either binary (1/0) or float (e.g., 10.24)

        # print(self.feat_dict.numeric_cols)
        # if 'c_indices' in self.feat_dict.numeric_cols:
        #     print(" in ")

        numeric_Xv = dfi[self.feat_dict.numeric_cols].values.tolist()

        dfi.drop(self.feat_dict.numeric_cols, axis=1, inplace=True)



        dfv = dfi.copy()

        for col in dfi.columns:

            if col in self.feat_dict.ignore_cols:

                dfi.drop(col, axis=1, inplace=True)

                dfv.drop(col, axis=1, inplace=True)

                continue

            else:

                dfi[col] = dfi[col].map(self.feat_dict.feat_dict[col])

                dfv[col] = 1.



        # list of list of feature indices of each sample in the dataset

        cate_Xi = dfi.values.tolist()

        # list of list of feature values of each sample in the dataset

        cate_Xv = dfv.values.tolist()

        if has_label:

            return cate_Xi, cate_Xv, numeric_Xv, y, ids

        else:

            return cate_Xi, cate_Xv, numeric_Xv, ids


def exp_(target):
    target = pd.to_numeric(target)
    target = pow(4.0 / 5.0, target)
    return target

def normalize01(target):
    target = pd.to_numeric(target)
    # print(target)
    target_min = target.min()
    target_max = target.max()

    target_normal = (target-target_min)/(target_max-target_min)
    return target_normal


# make sure the params are the same with the well trained model.
dfm_params = {

    "use_fm":True,

    "use_deep":True,

    "use_bn":True,

    "use_attention": True,

    "embedding_size":8,

    "dropout_fm":[1.0,1.0],

    "deep_layers":[512,256],

    "dropout_deep":[1.0,1.0,1.0],

    "deep_layer_activation":tf.nn.relu,

    "epoch":2,

    "mid_size":20000,

    "batch_size":2000,

    "learning_rate":0.0001,

    "optimizer":"adam",

    "batch_norm":1,

    "batch_norm_decay":0.995,

    "l2_reg":0.0,

    "verbose":True,

    "eval_metric":my_precision,

    "eval_metric_pair":evaluate_pair,

    "random_seed":config.RANDOM_SEED

}

## make new results dir
path = './New_drug _'+time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
print(path)

isExists = os.path.exists(path)


if not isExists:
    os.mkdir(path)
    print(path + ' 创建成功')
else:
    print(path + ' 目录已存在')


## data to predict

dfTest = pd.read_csv("data\GDSC_data_in_CCLE.csv") # GDSC common data

print(dfTest)

drug = np.unique(dfTest['d_indices'])
print(len(drug)) ## 47

dfTest.drop(["CName"],axis=1,inplace=True)
dfTest.rename(columns={"c_indices_GDSC":"c_indices"},inplace=True) ## dfTest 保留 GDSC_indices

dfTest_ = dfTest.drop(["c_indices"],axis=1, inplace=False)
dfTest_.rename(columns={"c_indices_CCLE":"c_indices"},inplace=True) ## dfTest_ 保留 CCLE_indices
print(dfTest_)


dfTest_.insert(0, 'c_indices', dfTest_.pop('c_indices'))
dfTest_.insert(1, 'd_indices', dfTest_.pop('d_indices'))
dfTest_.insert(2, 'target', dfTest_.pop('target'))
dfTest_.insert(3, 'id', dfTest_.pop('id'))



dfTest['target'] = exp_(dfTest['target'])


fig = pd.read_csv("data/fingerprints.csv")
fig.rename(columns={"DRUG_ID":"d_indices"},inplace=True)
print(fig.shape) ##(228, 882)

exp = pd.read_csv(config.CELL_FILE)

ccle_drug = pd.read_csv(config.DRUG_FILE) # the well trained model on CCLE dataset , so we should use the origin CCLE drug data to get the dictionary.

fd = FeatureDictionary(ccle_drug,

                       numeric_cols=config.NUMERIC_COLS,

                       ignore_cols=config.IGNORE_COLS,

                       cate_cols=config.CATEGORICAL_COLS)


data_parser = DataParser(feat_dict=fd)

dfm_params["cate_feature_size"] = fd.feat_dim

dfm_params["field_size"] = len(config.CATEGORICAL_COLS)

dfm_params['numeric_feature_size'] = len(config.NUMERIC_COLS)



_get = lambda x,l:[x[i] for i in l]



best_ndcg_results_cv = np.zeros((config.NUM_SPLITS, 6), dtype=float)
best_percentile_new_results_cv = np.zeros((config.NUM_SPLITS, 6), dtype=float)
best_precision_results_cv = np.zeros((config.NUM_SPLITS, 6), dtype=float)



y_test_meta = np.zeros((dfTest.shape[0], 3), dtype=float)

y_test_meta[:, 0] = np.array(dfTest)[:, 0] 

y_test_meta[:, 1] = np.array(dfTest_)[:, 1]

y_test = np.zeros((3,dfTest.shape[0]), dtype=float)

for i in range(3):
    print("fold = %d" % i)

    dfm_params["fold"] = i

    dfm = DeepFM(**dfm_params)

    y_test_meta[:, 2] = dfm.predict_train(data_parser, exp, fig, dfTest_)

    print(y_test_meta)

    pd.DataFrame(y_test_meta).to_csv("new_drug_prediction/new_drugs_fold_%d_predict_results.csv"%i, index=False)

    y_test[i,:] = y_test_meta[:, 2].T


    for j, k in enumerate(config.list_k):
        best_precision_results_cv[i,j] = my_precision(np.array(dfTest)[:, 2], np.array(y_test_meta),
                                               k)
        best_ndcg_results_cv[i,j] = my_NDCG(np.array(dfTest)[:, 2], np.array(y_test_meta),
                                     k)
        best_percentile_new_results_cv[i,j] = my_Percentile_new(np.array(dfTest)[:, 2], np.array(y_test_meta),k)

    print(best_precision_results_cv[i])
    print(best_ndcg_results_cv[i])
    print(best_percentile_new_results_cv[i])

    performance_cv = np.zeros((3,6),dtype=float)

    performance_cv[0,:] = np.array(best_precision_results_cv[i])
    performance_cv[1,:]= np.array(best_ndcg_results_cv[i])
    performance_cv[2,:] = np.array(best_percentile_new_results_cv[i])

    pd.DataFrame(performance_cv).to_csv(path+"/new_drugs_fold_%d_performance.csv" % i, index=False)



## Test prediction result

average_array = pd.DataFrame(y_test).mean()

y_test_meta[:, 2]  = average_array.T

print(" The Score on test dataset predicted by three models:")
print(y_test_meta)

pd.DataFrame(y_test_meta).to_csv(path+"/new_drugs_average_predict_results.csv", index=False)

## anverage result
percision_df = pd.DataFrame(best_precision_results_cv).mean()
ndcg_df = pd.DataFrame(best_ndcg_results_cv).mean()
percentile_new_df = pd.DataFrame(best_percentile_new_results_cv).mean()

percision_df = percision_df.values.T
ndcg_df = ndcg_df.values.T
percentile_new_df = percentile_new_df.values.T

average_result = np.hstack((percision_df, ndcg_df, percentile_new_df))

print("The average Score on test dataset predicted by three models:")
print(average_result)

pd.DataFrame(average_result).to_csv(path+"/new_drugs_average_performance.csv", index=False)


