
import pandas as pd
import numpy as np
import os
from time import time

# 'du' in Chinese means Space

def count_du(temp_list):
    # unique_ = np.unique(np.array(temp_list))
    total_count = 0
    count = 0
    for i in range(len(temp_list)-1):
        for j in range(i+1,len(temp_list)):
            if temp_list[i] == temp_list[j]:
                print("temp1:%s,i=%d, temp2:%s,j=%d, Space：%d"%(temp_list[i],i,temp_list[j],j,(j-i-1)))
                total_count += j-i-1
                count +=1
    if count == 0:
        return 0
    else:
        return total_count*1.0/count


def count_du_target(temp_list,each):
    '''
    total_count: total space value
    count： pair count
    '''
    total_count = 0
    count = 0


    all_index = get_all_index(temp_list,each)

    if len(all_index) == 1:
        return 0,0
    else:
        # print(all_index)
        for i in range(len(all_index)-1):
            for j in range(i+1,len(all_index)):
                    total_count += all_index[j]-all_index[i]-1
                    if i == 0:
                        count +=j
        # print("total_count=%d,count=%d"%(total_count,count))
        return total_count, count

def count_du2(temp_list,pathway):
    '''
    total_count: total space value
    count : pair number
    '''
    # unique_ = np.unique(np.array(temp_list))
    total_count = 0
    count = 0
    for i in range(len(temp_list)-1):
        for j in range(i+1,len(temp_list)):
            if temp_list[i] == temp_list[j]:
                if pathway[i] == pathway[j]:
                    print("temp1:%s,i=%d, temp2:%s,j=%d, Space：%d"%(temp_list[i],i,temp_list[j],j,(j-i-1)))
                    total_count += j-i-1
                    count +=1
    if count == 0:
        return 0
    else:
        return total_count*1.0/count

def get_all_index(templist,target):
    index_list = []
    cont = 0
    for i in range(len(templist)):
        if templist[i] == target:
            index_list.append(i)

    return index_list



def Count_Space(dataset='CCLE',new=False):

    if dataset == 'CCLE':

        ## d_id -> d_name
        drug_target_ccle = pd.read_csv("data/drug_target_ccle.csv")
        ccle_all = pd.read_csv("data/ccle_actarea.csv")

        data = ccle_all
        drug_target = drug_target_ccle
    elif dataset == 'GDSC':

        cgp_all = pd.read_csv("data/original_cgp_data.csv")
        ## d_id -> d_name
        drug_target_gdsc = pd.read_csv("data/drug_target_gdsc.csv")

        data = cgp_all
        drug_target = drug_target_gdsc
    else:
        print(" only CCLE or GDSC can be counted Space value. Or you can change the code.")
        exit()

    groupbyC = data.groupby('c_indices')
    print(len(groupbyC))

    total_count = 0

    total = 0

    total_count_dict = {}

    count_dict = {}

    for c_id, group in groupbyC:
        # print(c_id)
        # print(group)

        cell_line_space = 0
        cell_line_count = 0

        group = pd.merge(group, drug_target, on='d_indices', how='left')
        group = group.sort_values(by="target", ascending=True)
        group = group.dropna(axis=0, how='any')

        target_node = group['Target'].values.tolist()
        unique_target = np.unique(np.array(target_node))
        # pathway_node = group['Pathway'].values.tolist()
        # print(target_node)
        # print(group)

        for each_target in unique_target:
            # print(each_target)
            total_temp, count_temp = count_du_target(target_node, each_target)


            if count_temp != 0:
                if each_target in total_count_dict.keys():
                    total_count_dict[each_target] += total_temp
                    count_dict[each_target] += count_temp
                else:
                    total_count_dict[each_target] = total_temp
                    count_dict[each_target] = count_temp

                cell_line_space += total_temp
                cell_line_count += count_temp

        # print("space = %d, count = %d"%(cell_line_space,cell_line_count))
        if cell_line_count != 0:
            total += cell_line_space *1.0/cell_line_count

    total = total/len(np.unique(data['c_indices']))
    print("dataset:%s：the average Space value of all cell line in this dataset = %f"%(dataset,total))


    for (k, v) in total_count_dict.items():
        total_count_dict[k] = total_count_dict[k] * 1.0 / count_dict[k]

    df = pd.DataFrame([(i, j) for i, j in total_count_dict.items()],
                      columns=['Target', 'Space'])
    df.to_csv(path+"/drug_Space_%s_%f.csv" %(dataset,total), index=False)



if __name__ == '__main__':
    '''
       This file is to caculate the Space value of each Dataset.
    '''

    ## make new results dir
    path = './Space_dataset_' + time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
    print(path)

    isExists = os.path.exists(path)

    if not isExists:
        os.mkdir(path)
        print(path + ' 创建成功')
    else:
        print(path + ' 目录已存在')

    Count_Space(dataset='GDSC')