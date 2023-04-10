import pandas as pd 
from sklearn.model_selection import train_test_split
import yaml
import argparse



def process_data(config):

    # Read params file
    params = yaml.safe_load(open(config))
    seed = params['base']['random_state']
    raw_dataset_path = params['path']['raw_dataset']
    test_ratio = params['split_data']['split_ratio']
    X_train_path = params['split_data']['X_train']
    y_train_path = params['split_data']['y_train']
    X_test_path = params['split_data']['X_test']
    y_test_path = params['split_data']['y_test']



    # Load in the data
    df = pd.read_csv(raw_dataset_path)

    # Split into train and test sections
    y = df.pop("quality")
    X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=test_ratio, random_state=seed)

    # save processed data
    X_train.to_csv(X_train_path,index=False)
    X_test.to_csv(X_test_path,index=False)
    y_train.to_csv(y_train_path,index=False)
    y_test.to_csv(y_test_path,index=False)


if __name__=="__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    process_data(config=parsed_args.config)
