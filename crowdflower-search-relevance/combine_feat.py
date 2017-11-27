
import os
import sys
import numpy as np
import pandas as pd
from scipy.sparse import hstack, vstack
from sklearn.base import BaseEstimator
from sklearn.datasets import dump_svmlight_file
import pickle

def identity(x):
    return x

class SimpleTransform(BaseEstimator):
    def __init__(self, transformer=identity):
        self.transformer = transformer

    def fit(self, X, y=None):
        return self

    def fit_transform(self, X, y=None):
        return self.transform(X)

    def transform(self, X, y=None):
        return self.transformer(X)


#### function to combine features
def combine_feat(feat_names, feat_path_name):
    n_runs = 3
    n_folds = 3
    feat_folder = "../data/crowdflower-search-relevance/Feat/solution"
    print("==================================================")
    print("Combine features...")
    
    ######################
    ## Cross-validation ##
    ######################
    print("For cross-validation...")
    ## for each run and fold
    for run in range(1,n_runs+1):
        ## use 33% for training and 67 % for validation
        ## so we switch trainInd and validInd
        for fold in range(1,n_folds+1):
            print("Run: %d, Fold: %d" % (run, fold))
            path = "%s/Run%d/Fold%d" % (feat_folder, run, fold)
            save_path = "%s/%s/Run%d/Fold%d" % (feat_folder, feat_path_name, run, fold)
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            for i,(feat_name,transformer) in enumerate(feat_names):

                ## load train feat
                feat_train_file = "%s/train.%s.feat.pkl" % (path, feat_name)
                with open(feat_train_file, "rb") as f:
                    x_train = pickle.load(f)
                if len(x_train.shape) == 1:
                    x_train.shape = (x_train.shape[0], 1)

                ## load valid feat
                feat_valid_file = "%s/valid.%s.feat.pkl" % (path, feat_name)
                with open(feat_valid_file, "rb") as f:
                    x_valid = pickle.load(f)
                if len(x_valid.shape) == 1:
                    x_valid.shape = (x_valid.shape[0], 1)

                ## align feat dim
                dim_diff = abs(x_train.shape[1] - x_valid.shape[1])
                if x_valid.shape[1] < x_train.shape[1]:
                    x_valid = hstack([x_valid, np.zeros((x_valid.shape[0], dim_diff))]).tocsr()
                elif x_valid.shape[1] > x_train.shape[1]:
                    x_train = hstack([x_train, np.zeros((x_train.shape[0], dim_diff))]).tocsr()

                ## apply transformation
                x_train = transformer.fit_transform(x_train)
                x_valid = transformer.transform(x_valid)

                ## stack feat
                if i == 0:
                    X_train, X_valid = x_train, x_valid
                else:
                    try:
                        X_train, X_valid = hstack([X_train, x_train]), hstack([X_valid, x_valid])
                    except:
                        X_train, X_valid = np.hstack([X_train, x_train]), np.hstack([X_valid, x_valid])

                print("Combine {:>2}/{:>2} feat: {} ({}D)".format(i+1, len(feat_names), feat_name, x_train.shape[1]))
            print("Feat dim: {}D".format(X_train.shape[1]))

            ## load label
            # train
            info_train = pd.read_csv("%s/train.info" % (save_path))
            ## change it to zero-based for multi-classification in xgboost
            Y_train = info_train["median_relevance"] - 1 
            # valid               
            info_valid = pd.read_csv("%s/valid.info" % (save_path))
            Y_valid = info_valid["median_relevance"] - 1

            ## dump feat
            dump_svmlight_file(X_train, Y_train, "%s/train.feat" % (save_path))
            dump_svmlight_file(X_valid, Y_valid, "%s/valid.feat" % (save_path))
    
    ##########################
    ## Training and Testing ##
    ##########################
    print("For training and testing...")
    path = "%s/All" % (feat_folder)
    save_path = "%s/%s/All" % (feat_folder, feat_path_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    for i,(feat_name,transformer) in enumerate(feat_names):

        ## load train feat
        feat_train_file = "%s/train.%s.feat.pkl" % (path, feat_name)
        with open(feat_train_file, "rb") as f:
            x_train = pickle.load(f)
        if len(x_train.shape) == 1:
            x_train.shape = (x_train.shape[0], 1)

        ## load test feat
        feat_test_file = "%s/test.%s.feat.pkl" % (path, feat_name)
        with open(feat_test_file, "rb") as f:
            x_test = pickle.load(f)
        if len(x_test.shape) == 1:
            x_test.shape = (x_test.shape[0], 1)

        ## align feat dim
        dim_diff = abs(x_train.shape[1] - x_test.shape[1])
        if x_test.shape[1] < x_train.shape[1]:
            x_test = hstack([x_test, np.zeros((x_test.shape[0], dim_diff))]).tocsr()
        elif x_test.shape[1] > x_train.shape[1]:
            x_train = hstack([x_train, np.zeros((x_train.shape[0], dim_diff))]).tocsr()

        ## apply transformation
        x_train = transformer.fit_transform(x_train)
        x_test = transformer.transform(x_test)

        ## stack feat
        if i == 0:
            X_train, X_test = x_train, x_test
        else:
            try:
                X_train, X_test = hstack([X_train, x_train]), hstack([X_test, x_test])
            except:
                X_train, X_test = np.hstack([X_train, x_train]), np.hstack([X_test, x_test])

        print("Combine {:>2}/{:>2} feat: {} ({}D)".format(i+1, len(feat_names), feat_name, x_train.shape[1]))
    
    print("Feat dim: {}D".format(X_train.shape[1]))
    
    ## load label
    # train
    info_train = pd.read_csv("%s/train.info" % (save_path))
    ## change it to zero-based for multi-classification in xgboost
    Y_train = info_train["median_relevance"] - 1 
    # test               
    info_test = pd.read_csv("%s/test.info" % (save_path))
    Y_test = info_test["median_relevance"] - 1

    ## dump feat
    dump_svmlight_file(X_train, Y_train, "%s/train.feat" % (save_path))
    dump_svmlight_file(X_test, Y_test, "%s/test.feat" % (save_path))
    
    