import os
import sys
import numpy as np
import pandas as pd
import pickle

def gen_info(feat_path_name):
    data_folder = "../data/crowdflower-search-relevance"
    stratified_label = "query"
    feat_folder = "../data/crowdflower-search-relevance/Feat/solution"
    n_runs = 3
    processed_train_data_path = "../data/crowdflower-search-relevance/Feat/solution/train.processed.csv.pkl"
    processed_test_data_path = "../data/crowdflower-search-relevance/Feat/solution/test.processed.csv.pkl"
    original_train_data_path = "../data/crowdflower-search-relevance/train.csv"
    original_test_data_path = "../data/crowdflower-search-relevance/test.csv"
    
    ###############
    ## Load Data ##
    ###############
    ## load data
    with open(processed_train_data_path, "rb") as f:
        dfTrain = pickle.load(f)
    with open(processed_test_data_path, "rb") as f:
        dfTest = pickle.load(f)
        
    dfTrain_original = pd.read_csv(original_train_data_path).fillna("")
    dfTest_original = pd.read_csv(original_test_data_path).fillna("")
    ## insert fake label for test
    dfTest_original["median_relevance"] = np.ones((dfTest_original.shape[0]))
    dfTest_original["relevance_variance"] = np.zeros((dfTest_original.shape[0]))
    # change it to zero-based for classification
    Y = dfTrain_original["median_relevance"].values - 1
    ## load pre-defined stratified k-fold index
    with open("%s/stratifiedKFold.%s.pkl" % (data_folder, stratified_label), "rb") as f:
        skf = pickle.load(f)
        
     #######################
    ## Generate Features ##
    #######################
    print("Generate info...")
    print("For cross-validation...")
    for run in range(n_runs):
        ## use 33% for training and 67 % for validation
        ## so we switch trainInd and validInd
        for fold, (validInd, trainInd) in enumerate(skf[run]):
            print("Run: %d, Fold: %d" % (run+1, fold+1))
            path = "%s/%s/Run%d/Fold%d" % (feat_folder, feat_path_name, run+1, fold+1)
            if not os.path.exists(path):
                os.makedirs(path)
            ##########################
            ## get and dump weights ##
            ##########################
            raise_to = 0.5
            var = dfTrain["relevance_variance"].values
            max_var = np.max(var[trainInd]**raise_to)
            weight = (1 + np.power(((max_var - var**raise_to) / max_var),1)) / 2.
            #weight = (max_var - var**raise_to) / max_var
            np.savetxt("%s/train.feat.weight" % path, weight[trainInd], fmt="%.6f")
            np.savetxt("%s/valid.feat.weight" % path, weight[validInd], fmt="%.6f")    
            #############################    
            ## get and dump group info ##
            #############################
            np.savetxt("%s/train.feat.group" % path, [len(trainInd)], fmt="%d")
            np.savetxt("%s/valid.feat.group" % path, [len(validInd)], fmt="%d")
            ######################
            ## get and dump cdf ##
            ######################
            hist = np.bincount(Y[trainInd])
            overall_cdf_valid = np.cumsum(hist) / float(sum(hist))
            np.savetxt("%s/valid.cdf" % path, overall_cdf_valid)
            #############################
            ## dump all the other info ##
            #############################
            dfTrain_original.iloc[trainInd].to_csv("%s/train.info" % path, index=False, header=True)
            dfTrain_original.iloc[validInd].to_csv("%s/valid.info" % path, index=False, header=True)
    print("Done.")
    print("For training and testing...")
    path = "%s/%s/All" % (feat_folder, feat_path_name)
    if not os.path.exists(path):
        os.makedirs(path)
    ## weight
    max_var = np.max(var**raise_to)
    weight = (1 + np.power(((max_var - var**raise_to) / max_var),1)) / 2.
    np.savetxt("%s/train.feat.weight" % path, weight, fmt="%.6f")
    
    ## group
    np.savetxt("%s/%s/All/train.feat.group" % (feat_folder, feat_path_name), [dfTrain.shape[0]], fmt="%d")
    np.savetxt("%s/%s/All/test.feat.group" % (feat_folder, feat_path_name), [dfTest.shape[0]], fmt="%d")
    ## cdf
    hist_full = np.bincount(Y)
    print((hist_full) / float(sum(hist_full)))
    overall_cdf_full = np.cumsum(hist_full) / float(sum(hist_full))
    np.savetxt("%s/%s/All/test.cdf" % (feat_folder, feat_path_name), overall_cdf_full)
    ## info        
    dfTrain_original.to_csv("%s/%s/All/train.info" % (feat_folder, feat_path_name), index=False, header=True)
    dfTest_original.to_csv("%s/%s/All/test.info" % (feat_folder, feat_path_name), index=False, header=True)
    
    print("All Done.")