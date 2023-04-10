import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import yaml
import argparse
import json



def train(config):
        
        params = yaml.safe_load(open(config))

        
        X_train_path = params['split_data']['X_train']
        y_train_path = params['split_data']['y_train']
        X_test_path = params['split_data']['X_test']
        y_test_path = params['split_data']['y_test']
        seed = params['base']['random_state']
        scores_file = params["reports"]["scores"]
        params_file = params["reports"]["params"]
        max_depth_ = params['estimators']['max_depth']

        X_train=pd.read_csv(X_train_path)
        X_test=pd.read_csv(X_test_path)
        y_train=pd.read_csv(y_train_path)
        y_test=pd.read_csv(y_test_path)

        # Fit a model on the train section
        regr = RandomForestRegressor(max_depth=max_depth_, random_state=seed)
        regr.fit(X_train, y_train.values.ravel())

        # Report training set score
        train_score = regr.score(X_train, y_train) * 100
        # Report test set score
        test_score = regr.score(X_test, y_test) * 100

        
        # Dump output scores
        with open(scores_file, "w") as f:
                scores = {
                "Train score": train_score,
                "Test score": test_score
                }
                json.dump(scores, f, indent=4)

        # Dump parameters
        with open(params_file, "w") as f:
                params = {
                "max_depth": max_depth_,
                "seed": seed
                }
                json.dump(params, f, indent=4)



if __name__=="__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    train(config=parsed_args.config)   

