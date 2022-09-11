from pprint import pprint
import dvc.api

# from sklearn.ensemble import RandomForestRegressor
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_squared_error, r2_score

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import os
import sys
import matplotlib.pyplot as plt
from datetime import datetime

from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, PowerTransformer, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# import mlflow
# import mlflow.sklearn
from pickle import load

import io

# path_parent = os.path.dirname(os.getcwd())
# os.chdir(path_parent)
# sys.path.insert(0, path_parent+'/scripts')

print(os.getcwd())
path = "data/sample.csv"
# repo = '../'
version = "V1.0"

# data_url = dvc.api.read(path=path,
#                         repo=repo,
#                         rev=version
#                         )

mlflow.set_experiment('sales_prediction_bot')

def eval_metrics(actual, pred, verbose=True):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    if verbose:
        print("\tRMSE Score is: {:.5%}".format(rmse))
        print("\tR2 Square Score is: {:.5%}".format(r2))
        print("\tMAE Score is: {:.5%}".format(mae))
    return {f'RMSE Score': rmse, f'R2_Squared': r2, f'MAE Score': mae}


def main():
    # prepare example dataset
    data = pd.read_csv(path)
    with mlflow.start_run() as run:
        
        mlflow.log_param('data_version', version)
        mlflow.log_param('input_rows', data.shape[0])
        mlflow.log_param('input_colums', data.shape[1])
        data.StateHoliday = data.StateHoliday.astype('string')
        if 'Unnamed: 0' in data.columns:
            data.drop(columns=['Unnamed: 0'], inplace=True,axis=1)

        prep = load(open('models/data_transformer.pkl', 'rb'))
        
        X_train, X_test, y_train, y_test = train_test_split(data.drop(['Sales'],axis=1), data['Sales'], test_size=0.2, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

        X_train_trans = prep.transform(X_train)
        X_test_trans = prep.transform(X_test)
        X_val_trans = prep.transform(X_val)
                
        
        rf = RandomForestRegressor(bootstrap=True,max_depth=100,random_state=42,max_features=3,min_samples_leaf=3,min_samples_split=8,n_estimators=1000)
        rf.fit(X_train_trans, y_train)
        
        

        train_score = rf.score(X_train_trans, y_train)
        valid_score = rf.score(X_val_trans, y_val)
        
                
        # score = lr.score(X_test, y_test)

        # print("Score: %s" % score)
        mlflow.log_metric("train_score", train_score)
        mlflow.log_metric("test_score", test_score)
        mlflow.sklearn.log_model(rf, "model")
        print("Model saved in run %s" % mlflow.active_run().info.run_uuid)
        
        
        date = datetime.now()
        date = date.strftime("%A-%B-%Y : %I-%M-%S %p")
        
        
        
        rm_val, r2_val, mae_val = eval_metrics(y_val, rf.predict(X_val_trans))
        rm_test, r2_test, mae_test = eval_metrics(y_test, rf.predict(X_test_trans))
        
        with open('results.txt', 'w') as file:
            file.write('Random Forest Regressor Algorithm\n')
            file.write(f'Date:\n\t{date}\n')
            file.write('Validation Metrics:\n')
            file.write(f'RMSE:\n\t{rm_val}\n')
            file.write(f'R2Error:\n\t{r2_val}\n')
            file.write(f'MAE:\n\t{mae_val}\n')
            file.write('Testing Metrics:\n')
            file.write(f'RMSE:\n\t{rm_test}\n')
            file.write(f'R2Error:\n\t{r2_test}\n')
            file.write(f'MAE:\n\t{mae_test}\n')
        

if __name__ == "__main__":
    main()
