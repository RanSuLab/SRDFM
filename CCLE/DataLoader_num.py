

import numpy as np

import pandas as pd

import config

import random


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
                us = df[col].unique()

                self.feat_dict[col] = dict(zip(us, range(tc, len(us) + tc)))

                tc += len(us)

        self.feat_dim = tc


class DataParser(object):

    def __init__(self, feat_dict):

        self.feat_dict = feat_dict

    def parse(self, infile=None, df=None, has_label=False):

        assert not ((infile is None) and (df is None)
                    ), "infile or df at least one is set"

        assert not (
            (infile is not None) and (
                df is not None)), "only one can be set"

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

            dfi.drop(["id", "target", "c_indices", "d_indices"],
                     axis=1, inplace=True)

        else:

            id_cols = ['c_indices', 'd_indices', 'id']

            ids = dfi[id_cols].values.tolist()

            dfi.drop(["id", "target", "c_indices", "d_indices"],
                     axis=1, inplace=True)

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


# abort
def gen_pairs(y, ids):
    # print("gen_pairs start: *******************")

    pairs = []
    for i in range(0, len(ids)):
        # print("pair****:")
        # print(ids[i])
        for j in range(i + 1, len(ids)):
            # Only look at queries with the same id
            # print(ids[j])
            if (ids[i][0] != ids[j][0]):
                continue
            # Document pairs found with different rating
            if (ids[i][0] == ids[j][0] and ids[i][1] != ids[j][1]):
                # Sort by saving the largest index in position 0
                if (y[i] > y[j]):
                    pairs.append([ids[i][0], ids[i][3], ids[j][3]])
                else:
                    pairs.append([ids[i][0], ids[j][3], ids[i][3]])
    # print(pairs)
    return pairs


def gen_pairs_(y_id):

    pairs = []
    for i in range(0, len(y_id)):
        # print("pair****:")
        # print(ids[i])
        for j in range(i + 1, len(y_id)):
            # Only look at queries with the same id
            # print(ids[j])
            if (y_id[i][0] != y_id[j][0]):
                continue
            # Document pairs found with different rating
            if (y_id[i][0] == y_id[j][0] and y_id[i][1] != y_id[j]
                    [1]):  # the same cell-id with different drug_id
                # Sort by saving the largest index in position 0
                # print("y[i]:")
                # print("y_id[i][2] : %f, y_id[j][2] : %f" %y_id[i][2]%y_id[j][2])
                if (y_id[i][2] > y_id[j][2]):
                    # ids[i][0] is cell-id
                    pairs.append(
                        [y_id[i][0], int(y_id[i][3]), int(y_id[j][3])])
                else:
                    pairs.append(
                        [y_id[i][0], int(y_id[j][3]), int(y_id[i][3])])
    # print(pairs)
    return pairs


# abort
def gen_pairs_by_cell_line(train_array):
    '''
    for each_cell in cell lines:
        generate pairs one by one
    '''
    pairs = []

    y_id_train_df = pd.DataFrame(train_array, columns=config.cols)
    c_train = np.unique(y_id_train_df['c_indices'])
    print(len(c_train))
    K = config.K  # each drug will be used only K times in all pairs

    for each in c_train:
        # print("c_indices: %d"%each)
        each_df = y_id_train_df[y_id_train_df['c_indices'] == each]
        each_array = np.array(each_df)
        # print(each_array)

        drugs_index = []
        for i in range(0, len(each_df) - 1):
            drugs_index.append(i)

        d_dict = {}
        d_count = {}
        undo_drug = []

        for i in range(0, len(each_array) - 1):
            each_drug = int(each_array[i][1])
            # print(each_drug)
            # copy，will not change the original array.
            selective_drugs_index = drugs_index[:]
            count = 0
            selective_drugs_index.remove(i)
            # selective_drugs_index = random.sample(selective_drugs_index, min(K,len(selective_drugs_index)))
            # for j in selective_drugs_index:
            while count < K:
                if len(selective_drugs_index) == 0:
                    break

                j = random.sample(selective_drugs_index, 1)[0]
                if j > i:

                    if j not in undo_drug:

                        if (each_array[i][2] > each_array[j][2]):
                            pairs.append([each_array[i][0], int(each_array[i][3]),
                                          int(each_array[j][3])])  # ids[i][0] 是cell-id  i,j 是针对 ids的序号，
                        else:
                            pairs.append([each_array[i][0], int(
                                each_array[j][3]), int(each_array[i][3])])
                        if i not in d_dict.keys():
                            d_dict[i] = [j]
                        else:
                            temp = d_dict[i]
                            temp.append(j)
                            d_dict[i] = temp
                        count += 1

                        # 计入d_count
                        if j not in d_count.keys():
                            d_count[j] = 1
                        else:
                            d_count[j] += 1
                            if d_count[j] == len(drugs_index) - K - 1:
                                undo_drug.append(j)
                    else:
                        selective_drugs_index.remove(j)
                        continue

                else:
                    use_index = d_dict[j]
                    if i in use_index:
                        selective_drugs_index.remove(j)
                        continue
                    else:
                        if (each_array[i][2] > each_array[j][2]):
                            pairs.append([each_array[i][0], int(each_array[i][3]), int(
                                each_array[j][3])])  # ids[i][0] is cell-id
                        else:
                            pairs.append([each_array[i][0], int(
                                each_array[j][3]), int(each_array[i][3])])

                        if i not in d_dict.keys():
                            d_dict[i] = [j]
                        else:
                            temp = d_dict[i]
                            temp.append(j)
                            d_dict[i] = temp
                        count += 1

                selective_drugs_index.remove(j)

                # print(len(selective_drugs_index))

    return pairs


# abort
def gen_pairs_local(y_id):
    # print("gen_pairs start: *******************")
    pairs = []

    center_dict = {}  # key: center_cell, value: center_drug

    for i in range(0, len(y_id)):
        # print("pair****:")
        # print(ids[i])

        # regard the cell-line, which firstly showed up, as center
        center_cell = y_id[i][0]
        center_drug = y_id[i][1]
        if center_cell not in center_dict.keys():
            center_dict[center_cell] = center_drug
        else:
            # center cell already used, can be a part of another pairs
            continue

        for j in range(i + 1, len(y_id)):
            # Only look at queries with the same id
            # print(ids[j])
            if (y_id[i][0] != y_id[j][0]):
                continue
            # Document pairs found with different rating
            if (y_id[i][0] == y_id[j][0] and y_id[i][1] != y_id[j][1]):
                # Sort by saving the largest index in position 0
                # print("y[i]:")
                # print("y_id[i][2] : %f, y_id[j][2] : %f" %y_id[i][2]%y_id[j][2])
                if (y_id[i][2] > y_id[j][2]):

                    pairs.append([y_id[i][0], int(i), int(j)])

                else:
                    pairs.append([y_id[i][0], int(j), int(i)])

    # print(pairs)
    return pairs
