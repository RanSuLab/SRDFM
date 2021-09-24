# SRDFM


# Presentation

In this study, we designed a model, named SRDFM, to rank drugs and recommend the most effective drug based on the gene expression profile of cancer cell lines and chemical structure of anti-cancer drugs.

# Dataset

Cancer Cell Line Encyclopedia (CCLE)

Genomics of Drug Sensitivity in Cancer (GDSC)

Drug combination dataset

# CCLE
cell.csv: CCLE gene expression file

ccle_actarea.csv: CCLE actarea value file

drug.csv: CCLE fingerprint file

CName_cid_CCLE.csv: cell line cname-c_id file

# GDSC
cgp_cell_data.csv: GDSC gene expression file

original_cgp_data.csv: GDSC logIC50 value file

fingerprints.csv: GDSC fingerprint file

CName_cid_GDSC.csv: cell line cname-c_id file



# Version

Python 3.7

Tensorflow-gpu 1.14.0

pandas 0.25.3

numpy 1.17.4

sklearn 0.0


# Method
For CCLE and GDSC and Drug_combination

set different params in dfm_params and run main_num_pair.py to train. Then after a long time, we can obtain the best performance model.





