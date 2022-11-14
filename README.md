<h1 align="center">Pharmaceutical Company Sales Prediction</h1>
<div>
<a href="https://github.com/isaaclucky/sales-prediction/network/members"><img src="https://img.shields.io/github/forks/isaaclucky/sales-prediction" alt="Forks Badge"/></a>
<a href="https://github.com/isaaclucky/sales-prediction/pulls"><img src="https://img.shields.io/github/issues-pr/isaaclucky/sales-prediction" alt="Pull Requests Badge"/></a>
<a href="https://github.com/isaaclucky/sales-prediction/issues"><img src="https://img.shields.io/github/issues/isaaclucky/sales-prediction" alt="Issues Badge"/></a>
<a href="https://github.com/isaaclucky/sales-prediction/graphs/contributors"><img alt="GitHub contributors" src="https://img.shields.io/github/contributors/isaaclucky/sales-prediction?color=2b9348"></a>
<a href="https://github.com/isaaclucky/sales-prediction/blob/main/LICENSE"><img src="https://img.shields.io/github/license/isaaclucky/sales-prediction?color=2b9348" alt="License Badge"/></a>
</div>


</br>
<img src="images/pharma.jpg" name="">


## Data Exploration Link
- [Visualization link] (https://share.streamlit.io/]

## Articles
- [Medium Article](https://medium.com/@Yishak_Tadele/sales-prediction-bc1d89eb6807)

## Table of Contents

* [Pharmaceutical Company Sales Prediction](#sales-prediction)

  - [Introduction](##Introduction)
  - [Project Structure](#project-structure)
    * [data](#data)
    * [models](#models)
    * [notebooks](#notebooks)
    * [scripts](#scripts)
    * [sql](#sql)
    * [tests](#tests)
    * [logs](#logs)
    * [root folder](#root-folder)
  - [Installation guide](#installation-guide)

## Introduction

<img src="images/slide/" name="">
<img src="images/slide/" name="">

## Project Structure

### images:

- `images/` the folder where all snapshot for the project are stored.

### logs:

- `logs/` the folder where script logs are stored.

### mlruns:
- `mlruns/0/` the folder that contain auto generated mlflow runs.
### data:

 - `train_store.csv.dvc` the folder where the dataset versioned csv files are stored.

### .dvc:
- `.dvc/`: the folder where dvc is configured for data version control.

### .github:

- `.github/`: the folder where github actions and CML workflow is integrated.

### models:
- `%.pkl`: the folder where model pickle files are stored.

### notebooks:

- `data_preprocessing.ipynb`: a jupyter notebook for preprocessing the data.
- `data_exploration.ipynb`: a jupyter notebook for exploring the data.
- `ml_preprocess`: a jupyter notebook for preprocessing the data for ml analysis.
- `ml_model`: a jupyter notebook training an Regression models for prediction purpose.
- `time_series_ltsm.ipynb`: a jupyter notebook training an LSTM model for forecasting purpose.

###  scripts:

- `applications/`: folder where dashboard design are stored.
- `web-css/`: folder where style sheet are stored
      - `index.html`
      - `styles.css`
- `data_cleaner.py`: a python script for cleaning pandas dataframes.
- `data_preprocessing.py`:  a python script for accessing information data from a pandas dataframe.
- `ltsm_model`: a python script for model manipulation.
- `data_manipulator.py`: a python script for manipulating dataframes.
- `data_exploration.py`: a python script for plotting dataframes.
- `multiapp.py`: a python script for creating a multipaged streamlit app.
- `log_help.py`: a python script that creates python based logger.
### tests:

- `tests/`: the folder containing unit tests for the scripts.

### sql:

- `sql/`: the folder containing database table and mysql-python manipulator script.
### root folder

- `train.py`: holds cml report and model metrics.
- `results.txt`: contains cml pr reports.
- `requirements.txt`: a text file lsiting the projet's dependancies.
- `.travis.yml`: a configuration file for Travis CI for unit test.
- `app.py`: main file for the streamlit application.
- `setup.py`: a configuration file for installing the scripts as a package.
- `README.md`: Markdown text with a brief explanation of the project and the repository structure.
- `Dockerfile`: build users can create an automated build that executes several command-line instructions in a container.

## Installation guide

```bash
git clone https://github.com/isaaclucky/sales-prediction.git
cd sales-prediction
sudo python3 setup.py install
```
