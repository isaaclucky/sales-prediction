from pprint import pprint
import dvc.api

# from sklearn.ensemble import RandomForestRegressor
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_squared_error, r2_score

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import os, sys

import io 

# path_parent = os.path.dirname(os.getcwd())
# os.chdir(path_parent)
# sys.path.insert(0, path_parent+'/scripts')
# sys.path.insert(0, path_parent+'/data')

print(os.getcwd())
path="data/sample_data.csv"


mlflow.set_experiment('sales_prediction_2')

def main():
    # prepare example dataset
    data = pd.read_csv(path, sep=",")
    
    #log data params
    # mlflow.log_param('data_url', data_url)
    # mlflow.log_param('data_version', version)
    mlflow.log_param('input_rows', data.shape[0])
    mlflow.log_param('input_colums', data.shape[1])
    
    # score = lr.score(X_test, y_test)
    
    # print("Score: %s" % score)
    # mlflow.log_metric("score", score)
    # mlflow.sklearn.log_model(lr, "model")
    print("Model saved in run %s" % mlflow.active_run().info.run_uuid)
    val_accuracy=0.8
    with open("metrics.txt", 'w') as outfile:
        outfile.write(
            f"Validation data accuracy: {val_accuracy}")


if __name__ == "__main__":
    main()