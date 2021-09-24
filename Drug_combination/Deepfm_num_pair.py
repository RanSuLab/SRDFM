import csv

import numpy as np
import pandas as pd

import tensorflow as tf

import gc

from time import time

from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.metrics import roc_auc_score

from DataLoader_num import gen_pairs, gen_pairs_,gen_pairs_hard

import config2 as config

from metrics import my_precision,my_Percentile,my_NDCG,my_Percentile_new

# from cosine_loss import cosine_loss

class DeepFM(BaseEstimator, TransformerMixin):



    def __init__(self, cate_feature_size, field_size, numeric_feature_size,

                 embedding_size=8, dropout_fm=[1.0, 1.0],

                 deep_layers=[32, 32], dropout_deep=[0.5, 0.5, 0.5],

                 deep_layer_activation=tf.nn.relu,

                 epoch=1, batch_size=1024, mid_size = 40000,

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
            np.random.seed(self.random_seed)



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




            if self.use_attention:
                self.attention_o1 = tf.add(tf.matmul(tf.reshape(self.embeddings_o1,shape=[-1,self.field_size * self.embedding_size]), self.weights["attention_w"]),
                                           self.weights["attention_b"])

                self.attention_o1 = tf.sigmoid(self.attention_o1)  # 0-1

                print(self.attention_o1)

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
                #
                print(self.x0_o1.shape)  # N * 26036
                #
                self.x0_o1 = tf.layers.batch_normalization(self.x0_o1, training=self.train_phase)




            self.embeddings_o2 = tf.nn.embedding_lookup(self.weights['feature_embeddings'],self.feat_index_o2) # N * 881 * 8

            feat_value_o2 = tf.reshape(self.feat_value_o2,shape=[-1,self.field_size,1]) # N * 881 *1

            self.embeddings_o2 = tf.multiply(self.embeddings_o2,feat_value_o2)  # N * 881 * 8



            if self.use_attention:

                self.attention_o2 = tf.add(tf.matmul( tf.reshape(self.embeddings_o2,shape=[-1,self.field_size * self.embedding_size]), self.weights["attention_w"]),
                                           self.weights["attention_b"])

                self.attention_o2 = tf.sigmoid(self.attention_o2)  # 0-1

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


            # BN
            if self.use_bn:
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

                with tf.control_dependencies(update_ops):
                    self.optimizer = self.optimizer.minimize(self.loss)
            else:
                self.optimizer = self.optimizer.minimize(self.loss)







            #init

            self.saver = tf.train.Saver()

            init = tf.global_variables_initializer()

            self.sess = tf.Session()

            self.sess.run(init)
            # self.sess.run(train_opt,feed_dict=grads_dict)






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


        # attention

        glorot = np.sqrt(2.0 / (self.field_size * self.embedding_size + self.numeric_feature_size))

        weights["attention_w"] = tf.Variable(
            np.random.normal(loc=0, scale=glorot, size=(self.field_size * self.embedding_size, self.numeric_feature_size)), dtype=np.float32)

        weights["attention_b"] = tf.Variable(np.random.normal(loc=0, scale=glorot, size=(1, self.numeric_feature_size)),
                                             dtype=np.float32)



        return weights




    def get_batch(self,pairs_train,batch_size,index):

        start = index * batch_size

        end = (index + 1) * batch_size

        end = end if end < len(pairs_train) else len(pairs_train)

        # return Xi[start:end],Xv[start:end],Xv2[start:end],[[y_] for y_ in y[start:end]]

        # print("get_batch:")
        #
        # print(y[start:end])

        return pairs_train[start:end]


    def get_batch_feature(self,Xi,Xv,Xv2,y, batch_size,index):

        start = index * batch_size

        end = (index + 1) * batch_size

        end = end if end < len(Xi) else len(Xi)

        return Xi[start:end],Xv[start:end],Xv2[start:end],[[y_] for y_ in y[start:end]]



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



    def shuffle_in_unison_scary(self, a):

        rng_state = np.random.get_state()

        np.random.seed(self.random_seed)

        np.random.shuffle(a)





    def evaluate(self, Xi, Xv, Xv2, y):

        """

        :param Xi: list of list of feature indices of each sample in the dataset

        :param Xv: list of list of feature values of each sample in the dataset

        :param y: label of each sample in the dataset

        :return: metric of the evaluation

        """


        y_pred = self.predict(Xi, Xv, Xv2)


        print("evaluate:")
        print(y_pred)
        # print(len(y))
        #
        # print(np.array(y)[:,2])
        #
        # datasize = len(Xi_o1)
        return self.eval_metric(y, y_pred, 3)



    def evaluate_pair_train(self,data_parser,exp,fig,y_id):

        """

        :param Xi: list of list of feature indices of each sample in the dataset

        :param Xv: list of list of feature values of each sample in the dataset

        :param y: label of each sample in the dataset

        :return: metric of the evaluation

        """


        y_pred_o1, y_pred_o2, y_label_o1, y_label_o2 = self.predict_pair_train(data_parser, exp,fig,y_id)


        # print("evaluate_pair:")
        # print(y_pred_o1)
        # print(y_pred_o2)
        #### print valid_loss:



        return self.eval_metric_pair(y_pred_o1, y_pred_o2, y_label_o1, y_label_o2)




    def evaluate_pair(self,Xi, Xv, Xv2, y, pairs):

        """

        :param Xi: list of list of feature indices of each sample in the dataset

        :param Xv: list of list of feature values of each sample in the dataset

        :param y: label of each sample in the dataset

        :return: metric of the evaluation

        """


        y_pred_o1, y_pred_o2, y_label_o1, y_label_o2 = self.predict_pair(Xi, Xv, Xv2, y, pairs)


        print("evaluate_pair:")
        print(y_pred_o1)
        print(y_pred_o2)


        return self.eval_metric_pair(y_pred_o1, y_pred_o2, y_label_o1, y_label_o2)


    #
    def predict(self, Xi, Xv, Xv2):

        """
        ## predict() 函数不考虑 pair，直接取batch预测

        :param Xi: list of list of feature indices of each sample in the dataset

        :param Xv: list of list of feature values of each sample in the dataset

        :param ids_cd: [c_id, d_id]

        :return: predicted probability of each sample

        """


        dummy_y = [1] * len(Xi)

        print("test predict start!")



        batch_index = 0

        Xi_batch_o1, Xv_batch_o1, Xv2_batch_o1, y_batch_o1 = self.get_batch_feature(Xi, Xv, Xv2,dummy_y, self.batch_size, batch_index)

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

                         self.train_phase: False}

            batch_out_o1,weights,optimizer = self.sess.run([self.out_o1,self.weights['concat_projection'],self.optimizer], feed_dict=feed_dict)

            # print(optimizer)

            # print(weights)



            if batch_index == 0:

                y_pred_o1 = np.reshape(batch_out_o1, (num_batch,))



            else:

                y_pred_o1 = np.concatenate((y_pred_o1, np.reshape(batch_out_o1, (num_batch,))))





            # print("~~~~~~~~~~~~~~~~~y_pred:~~~~~~~~~~~~~~~~~~~~~")
            #
            # print(y_pred)



            batch_index += 1

            Xi_batch_o1, Xv_batch_o1, Xv2_batch_o1, y_batch_o1 = self.get_batch_feature(Xi, Xv, Xv2,dummy_y, self.batch_size, batch_index)



        return y_pred_o1


    def predict_train(self, data_parser, exp,fig,y_id):


        dummy_y = [1] * len(y_id) # 行数

        print("Un-Pair Predict start!")


        batch_index = 0

        Xi_batch_o1, Xv_batch_o1, Xv2_batch_o1, y_batch_o1 = self.get_batch_feature_train(data_parser, exp,fig,y_id, dummy_y,self.batch_size, batch_index)

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

                         self.train_phase: False}

            batch_out_o1,weights,optimizer = self.sess.run([self.out_o1,self.weights['concat_projection'],self.optimizer], feed_dict=feed_dict)

            # print(optimizer)

            # print(weights)



            if batch_index == 0:

                y_pred_o1 = np.reshape(batch_out_o1, (num_batch,))



            else:

                y_pred_o1 = np.concatenate((y_pred_o1, np.reshape(batch_out_o1, (num_batch,))))





            # print("~~~~~~~~~~~~~~~~~y_pred:~~~~~~~~~~~~~~~~~~~~~")
            #
            # print(y_pred)



            batch_index += 1

            # Xi_batch_o1, Xv_batch_o1, Xv2_batch_o1, y_batch_o1 = self.get_batch_feature(Xi, Xv, Xv2,dummy_y, self.batch_size, batch_index)
            Xi_batch_o1, Xv_batch_o1, Xv2_batch_o1, y_batch_o1 = self.get_batch_feature_train(data_parser, exp,fig,y_id, dummy_y,self.batch_size, batch_index)



        return y_pred_o1


    def predict_pair_train(self, data_parser, exp,fig,y_id):


        total_batch = int(len(y_id) / self.batch_size)
        print("************* total_batch = %d"%total_batch)


        y_pred_o1 = None

        y_pred_o2 = None

        y_label_o1 = None

        y_label_o2 = None

        total_pairs = 0

        for batch_index in range(total_batch):

            print("batch_index: %d"%batch_index)

            y_id_ = self.get_batch(y_id, self.batch_size, batch_index)

            pairs_batch = gen_pairs_(y_id_)
            print("pairs_batch count = %d" %len(pairs_batch))
            

            total_pairs += len(pairs_batch)

            self.shuffle_in_unison_scary(pairs_batch)


            o1_rows = np.array(pairs_batch)[:, 1].astype('int64')

            o2_rows = np.array(pairs_batch)[:, 2].astype('int64')


            # id_list
            id_list_o1 = np.array(y_id_)[o1_rows, 3].tolist()
            id_list_o2 = np.array(y_id_)[o2_rows, 3].tolist()


            df1 = pd.DataFrame(columns=["id"])
            df1["id"] = id_list_o1
            df2 = pd.DataFrame(columns=["id"])
            df2["id"] = id_list_o2

            y_id_train_df = pd.DataFrame(y_id_, columns=config.cols)
            
            batchTrain1 = pd.merge(y_id_train_df, df1, on="id", how="right")
            batchTrain2 = pd.merge(y_id_train_df, df2, on="id", how="right")

            # concat with exp、fig
            batchTrain1 = pd.merge(batchTrain1, exp, on='c_indices', how="left")
            batchTrain1 = pd.merge(batchTrain1, fig, on="d_indices", how="left")


            batchTrain2 = pd.merge(batchTrain2, exp, on='c_indices', how="left")
            batchTrain2 = pd.merge(batchTrain2, fig, on="d_indices", how="left")


            if len(batchTrain1) != len(id_list_o1):
                print("batchTrain load wrong!")
                input()

            cate_Xi_batch_o1, cate_Xv_batch_o1, numeric_Xv_batch_o1, ids = data_parser.parse(df=batchTrain1)

            cate_Xi_batch_o2, cate_Xv_batch_o2, numeric_Xv_batch_o2, ids = data_parser.parse(df=batchTrain2)

            del batchTrain1
            gc.collect()

            del batchTrain2
            gc.collect()
            
            del df1
            gc.collect()
            
            del df2
            gc.collect()
            
            del y_id_train_df
            gc.collect()



            y_batch_o1 = [[y_] for y_ in [1] * len(cate_Xi_batch_o1)]

            y_batch_o2 = [[y_] for y_ in [1] * len(cate_Xi_batch_o2)]

            num_batch = len(cate_Xi_batch_o1)

            feed_dict = {self.feat_index_o1: cate_Xi_batch_o1,

                         self.feat_value_o1: cate_Xv_batch_o1,

                         self.label_o1: y_batch_o1,

                         self.numeric_value_o1: numeric_Xv_batch_o1,

                         self.feat_index_o2: cate_Xi_batch_o2,

                         self.feat_value_o2: cate_Xv_batch_o2,

                         self.label_o2: y_batch_o2,

                         self.numeric_value_o2: numeric_Xv_batch_o2,

                         self.dropout_keep_fm: self.dropout_fm,

                         self.dropout_keep_deep: [1.0] * len(self.dropout_dep),

                         self.train_phase: False}

            batch_out_o1, batch_out_o2 = self.sess.run([self.out_o1, self.out_o2], feed_dict=feed_dict)

            batch_label_o1 = [[y_] for y_ in np.array(y_id_)[o1_rows, 2]]

            batch_label_o2 = [[y_] for y_ in np.array(y_id_)[o2_rows, 2]]

            # print(optimizer)

            # print(weights)

            if batch_index == 0:

                y_pred_o1 = np.reshape(batch_out_o1, (num_batch,))

                y_pred_o2 = np.reshape(batch_out_o2, (num_batch,))

                y_label_o1 = np.reshape(batch_label_o1, (num_batch,))

                y_label_o2 = np.reshape(batch_label_o2, (num_batch,))


            else:

                y_pred_o1 = np.concatenate((y_pred_o1, np.reshape(batch_out_o1, (num_batch,))))

                y_pred_o2 = np.concatenate((y_pred_o2, np.reshape(batch_out_o2, (num_batch,))))

                y_label_o1 = np.concatenate((y_label_o1, np.reshape(batch_label_o1, (num_batch,))))

                y_label_o2 = np.concatenate((y_label_o2, np.reshape(batch_label_o2, (num_batch,))))


        print("train集中 total pairs 数目：")
        print(total_pairs)



        return y_pred_o1, y_pred_o2, y_label_o1, y_label_o2




    def predict_pair(self, Xi, Xv, Xv2, y, pairs):

        """

        :param Xi: list of list of feature indices of each sample in the dataset

        :param Xv: list of list of feature values of each sample in the dataset

        :param ids_cd: [c_id, d_id]

        :return: predicted probability of each sample

        """


        # dummy_y = [1] * len(Xi)
        #
        # print("predict start!")




        batch_index = 0

        pairs_batch  = self.get_batch(pairs, self.batch_size, batch_index)

        y_pred_o1 = None

        y_pred_o2 = None

        y_label_o1 = None

        y_label_o2 = None


        while len(pairs_batch) > 0:
            o1_rows = np.array(pairs_batch)[:, 1]

            o2_rows = np.array(pairs_batch)[:, 2]



            cate_Xi_batch_o1 = np.array(Xi)[o1_rows, :]

            cate_Xi_batch_o2 = np.array(Xi)[o2_rows, :]

            cate_Xv_batch_o1 = np.array(Xv)[o1_rows, :]

            cate_Xv_batch_o2 = np.array(Xv)[o2_rows, :]

            numeric_Xv_batch_o1 = np.array(Xv2)[o1_rows, :]

            numeric_Xv_batch_o2 = np.array(Xv2)[o2_rows, :]

            # print(np.array(y_train))

            y_batch_o1 = [[y_] for y_ in [1] * len(cate_Xi_batch_o1)]

            y_batch_o2 = [[y_] for y_ in [1] * len(cate_Xi_batch_o1)]

            num_batch = len(cate_Xi_batch_o1)


            feed_dict = {self.feat_index_o1: cate_Xi_batch_o1,

                         self.feat_value_o1: cate_Xv_batch_o1,

                         self.label_o1: y_batch_o1,

                         self.numeric_value_o1: numeric_Xv_batch_o1,

                         self.feat_index_o2: cate_Xi_batch_o2,

                         self.feat_value_o2: cate_Xv_batch_o2,

                         self.label_o2: y_batch_o2,

                         self.numeric_value_o2: numeric_Xv_batch_o2,

                         self.dropout_keep_fm: self.dropout_fm,

                         self.dropout_keep_deep: [1.0] * len(self.dropout_dep),

                         self.train_phase: False}

            batch_out_o1,batch_out_o2 = self.sess.run([self.out_o1,self.out_o2], feed_dict=feed_dict)

            batch_label_o1 = [[y_] for y_ in np.array(y)[o1_rows, 2]]

            batch_label_o2 = [[y_] for y_ in np.array(y)[o2_rows, 2]]



            # print(optimizer)

            # print(weights)



            if batch_index == 0:

                y_pred_o1 = np.reshape(batch_out_o1, (num_batch,))

                y_pred_o2 = np.reshape(batch_out_o2, (num_batch,))

                y_label_o1 = np.reshape(batch_label_o1, (num_batch,))

                y_label_o2 = np.reshape(batch_label_o2, (num_batch,))



            else:

                y_pred_o1 = np.concatenate((y_pred_o1, np.reshape(batch_out_o1, (num_batch,))))

                y_pred_o2 = np.concatenate((y_pred_o2, np.reshape(batch_out_o2, (num_batch,))))

                y_label_o1 = np.concatenate((y_label_o1, np.reshape(batch_label_o1, (num_batch,))))

                y_label_o2 = np.concatenate((y_label_o2, np.reshape(batch_label_o2, (num_batch,))))





            # print("~~~~~~~~~~~~~~~~~y_pred:~~~~~~~~~~~~~~~~~~~~~")
            #
            # print(y_pred)



            batch_index += 1

            pairs_batch = self.get_batch(pairs, self.batch_size, batch_index)



        return y_pred_o1, y_pred_o2, y_label_o1, y_label_o2



    def fit_on_batch(self,Xi_o1,Xv_o1,Xv2_o1,y_o1,Xi_o2,Xv_o2,Xv2_o2,y_o2):

        t2 = time()

        feed_dict = {self.feat_index_o1:Xi_o1,

                     self.feat_value_o1:Xv_o1,

                     self.numeric_value_o1:Xv2_o1,

                     self.label_o1:y_o1,

                     self.feat_index_o2: Xi_o2,

                     self.feat_value_o2: Xv_o2,

                     self.numeric_value_o2: Xv2_o2,

                     self.label_o2: y_o2,

                     self.dropout_keep_fm:self.dropout_fm,

                     self.dropout_keep_deep:self.dropout_dep,

                     self.train_phase:True}



        loss, o12, h_o12,a, optimizer,weight = self.sess.run([self.loss, self.o12,self.h_o12,self.a,self.optimizer, self.weights['attention_w']], feed_dict=feed_dict)



        return loss



    def fit(self, data_parser, exp, fig, y_id_train, y_id_valid=None,

            early_stopping=False, refit=False, fold = 0):


        y_train_meta = np.zeros((len(y_id_train), 3), dtype=float)

        y_train_meta[:, 0] = np.array(y_id_train)[:, 0]

        y_train_meta[:, 1] = np.array(y_id_train)[:, 1]

        y_valid_meta = np.zeros((len(y_id_valid), 3), dtype=float)

        y_valid_meta[:, 0] = np.array(y_id_valid)[:, 0]  # c_indices

        y_valid_meta[:, 1] = np.array(y_id_valid)[:, 1]  # d_indices

        precision_results_cv = np.zeros(len(config.list_k), dtype=float)
        ndcg_results_cv = np.zeros(len(config.list_k), dtype=float)
        percentile_new_results_cv = np.zeros(len(config.list_k), dtype=float)

        best_precision = np.zeros(len(config.list_k), dtype=float)
        best_ndcg = np.zeros(len(config.list_k), dtype=float)
        best_new_percentile = np.zeros(len(config.list_k), dtype=float)
        best_valid_predict_gini = 0.0
        best_epoch = 0


        has_valid = y_id_valid is not None



        ### generate pairs in mid_size
        total_pairs = []

        if int(len(y_id_train) % self.mid_size) == 0:
            mid_count = int(len(y_id_train) / self.mid_size)
        else:
            mid_count = int(len(y_id_train) / self.mid_size) + 1

        print("mid_count:*************")
        print(mid_count)

        for i in range(mid_count):
            # print("mid_index: %d" % i)

            y_id_train_mid = self.get_batch(y_id_train, self.mid_size, i)
            pairs_mid = gen_pairs_(y_id_train_mid)

            total_pairs.extend(pairs_mid)

        print("pairs total count = %d" % len(total_pairs))

        self.shuffle_in_unison_scary(total_pairs)

        y_id_train_df = pd.DataFrame(y_id_train, columns=config.cols)



        print("self.epoch = %d"%self.epoch)

        for epoch in range(self.epoch):

            print("epoch = %d"%epoch)

            t1 = time()

            Total_loss = 0.0

            # self.shuffle_in_unison_scary(pairs_train)

            total_batch = int(len(total_pairs) / self.batch_size)
            print("total_batch = %d"%total_batch)


            for i in range(total_batch):
                print("batch = %d"%i)


                pairs_batch = self.get_batch(total_pairs, self.batch_size, i)


                ## obtain cate_Xi_batch_o1 ， cate_Xv_batch_o1 ，numeric_Xv_batch_o1
                # id_list
                id_list_o1 = np.array(pairs_batch)[:, 1].astype('int64') # ids_train_
                id_list_o2 = np.array(pairs_batch)[:, 2].astype('int64')

                df1 = pd.DataFrame(columns=["id"])
                df1["id"] = id_list_o1
                df2 = pd.DataFrame(columns=["id"])
                df2["id"] = id_list_o2

                
                batchTrain1 = pd.merge(y_id_train_df, df1, on="id", how="right")
                batchTrain2 = pd.merge(y_id_train_df, df2, on="id", how="right")

                #concat with exp、fig
                batchTrain1 = pd.merge(batchTrain1,exp,on='c_indices',how="left")
                batchTrain1 = pd.merge(batchTrain1,fig,on="d_indices",how="left")


                batchTrain2 = pd.merge(batchTrain2, exp, on='c_indices', how="left")
                batchTrain2 = pd.merge(batchTrain2, fig, on="d_indices", how="left")




                if len(batchTrain1) != len(id_list_o1):
                    print("Deepfm_num_pair.py #1620 batchTrain load wrong!")
                    exit()


                cate_Xi_batch_o1, cate_Xv_batch_o1, numeric_Xv_batch_o1,y_o1,ids = data_parser.parse(df=batchTrain1,has_label=True)

                cate_Xi_batch_o2, cate_Xv_batch_o2, numeric_Xv_batch_o2,y_o2,ids = data_parser.parse(df=batchTrain2,has_label=True)


                del batchTrain1
                gc.collect()

                del batchTrain2
                gc.collect()
                
                del df1
                gc.collect()
                
                del df2
                gc.collect()



                y_batch_o1 = np.array(y_o1)[:, 2].reshape((-1, 1))

                y_batch_o2 = np.array(y_o2)[:, 2].reshape((-1, 1))

                # loss
                loss = self.fit_on_batch(cate_Xi_batch_o1, cate_Xv_batch_o1,numeric_Xv_batch_o1, y_batch_o1, cate_Xi_batch_o2, cate_Xv_batch_o2,numeric_Xv_batch_o2, y_batch_o2)


                print("************** loss:  %f" % loss)

                Total_loss += loss

            print("train 中， Total——loss = %f" % Total_loss)

            Each_pair_loss = 1.0 * Total_loss / total_batch

            print("train 中，Average loss = %f" % Each_pair_loss)


            if has_valid:
                print("________________ evaluate_pair_valid _______________________")

                y_valid_meta[:, 2] = self.predict_train(data_parser, exp, fig, y_id_valid)

                for j, k in enumerate(config.list_k):
                    precision_results_cv[j] = my_precision(np.array(y_id_valid)[:, 2],np.array(y_valid_meta),k)
                    ndcg_results_cv[j] = my_NDCG(np.array(y_id_valid)[:, 2],np.array(y_valid_meta),k)
                    percentile_new_results_cv[j] = my_Percentile_new(np.array(y_id_valid)[:, 2],np.array(y_valid_meta),k)


                valid_predict_gini = precision_results_cv[1]  # k=3

                self.valid_result.append(valid_predict_gini)


                if valid_predict_gini > best_valid_predict_gini:
                    best_precision = precision_results_cv
                    best_ndcg = ndcg_results_cv
                    best_new_percentile = percentile_new_results_cv
                    best_valid_predict_gini = valid_predict_gini
                    best_epoch = epoch


            if self.verbose > 0 and epoch % self.verbose == 0:

                if has_valid:

                    print("[%d]  valid_precision = %.4f [%.1f s]"

                          % (epoch + 1, valid_predict_gini, time() - t1))

                else:

                    print("[%d] No valid_precision [%.1f s]"
                          %(epoch + 1 , time() - t1))


            if has_valid and early_stopping and self.training_termination(self.valid_result, best_gini=best_valid_predict_gini):

                break



        return best_precision, best_ndcg, best_new_percentile



    def training_termination(self, valid_result, best_gini):

        if len(valid_result) > 5:

            if self.greater_is_better:

                if valid_result[-1] < best_gini and valid_result[-2] < best_gini and valid_result[-3] < best_gini and valid_result[-4] < best_gini:

                    return True

            else:

                if valid_result[-1] > valid_result[-2] and valid_result[-2] > valid_result[-3] and valid_result[-3] > valid_result[-4] and valid_result[-4] > valid_result[-5]:

                    return True

        return False

