import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer
from log_supp import App_Logger

app_logger = App_Logger("../logs/data_cleaner.log").get_app_logger()


class Clean_df:
    def __init__(self, df: pd.DataFrame, deep=False) -> None:
        """
        Returns a DataCleaner Object with the passed DataFrame Data set as its own DataFrame
        Parameters
        ----------
        df:
            Type: pd.DataFrame
        Returns
        -------
        None
        """
        self.logger = App_Logger(
            "../logs/data_cleaner.log").get_app_logger()
        if(deep):
            self.df = df.copy(deep=True)
        else:
            self.df = df

    def get_numerical_columns(self) -> list:
        """
        Returns numerical column names
        """
        return self.df.select_dtypes(include='number').columns

    def drop_null_entries(self) -> pd.DataFrame:
        """
        Checks if there is a null entry in the dataset and removes them
        """
        self.df.dropna(subset=self.df.columns, axis=0, inplace=True)
        return self.df

    def convert_to_datetime(self, column: str) -> pd.DataFrame:
        """Convert column to datetime."""
        try:
            self.logger.info('Converting Column to Datetime')
            self.df[column] = pd.to_datetime(self.df[column])
            return self.df
        except Exception:
            self.logger.exception(
                'Failed to convert Column to Datetime')
            sys.exit(1)

    def label_encode(self, col_names: list) -> pd.DataFrame:
        """ Performs Label encoding of the given columns

        Parameters
        ------------
        df: Pandas DataFrame: dataframe to be computed
        Columns: list of columns
        Returns
        ------------
        The method returns a dataframe with label encoded categorical features
        """

        le = LabelEncoder()
        for col in col_names:
            self.df[col+'_l_encoded'] = le.fit_transform(self.df[col])

        self.df.drop(columns=col_names, axis=1, inplace=True)
        return self.df

    def one_hot_encode(self, col_names: list) -> pd.DataFrame:
        """ Performs One hot encoding of the given columns

        Parameters
        ------------
        df: Pandas DataFrame: dataframe to be computed
        Columns: list of columns
        Returns
        ------------
        The method returns a dataframe with One-hot encoded categorical features
        """
        # ohe = OneHotEncoder(handle_unknown='ignore')

        return pd.get_dummies(self.df, columns=col_names)

    def drop_duplicate(self) -> pd.DataFrame:
        """Drop duplicate rows."""
        self.logger.info('Dropping duplicate row')
        self.df = self.df.drop_duplicates(subset='Date')

        return self.df

    def convert_to_datetime(self, column: str) -> pd.DataFrame:
        """Convert column to datetime."""
        try:

            self.df[column] = pd.to_datetime(self.df[column])
            return self.df
        except Exception:
            self.logger.exception(
                'Failed to convert Column to Datetime')
            sys.exit(1)

    def minmax_scaling(self) -> pd.DataFrame:
        """
        Returns dataframe with minmax scaled columns
        """
        scaller = MinMaxScaler()
        res = pd.DataFrame(
            scaller.fit_transform(
                self.df[self.get_numerical_columns(self.df)]), columns=self.get_numerical_columns(self.df)
        )
        return res

    def fill_missing_with_zero(self, df, columns):
        """Fill missing data with zero."""
        try:
            # self.logger.info('Filling Missing Data with Zero')
            for col in columns:
                df[col] = df[col].fillna(0)
            return df
        except Exception:
            # self.logger.exception(
            #     'Failed to Fill Missing Data with Zero')
            sys.exit(1)
    def join_dataframes(self, df1, df2, on, how="inner"):
        """Join the two dataframes."""
        try:
            self.logger.info('Joining two Dataframes')
            return pd.merge(df1, df2, on=on)
        except Exception:
            self.logger.exception(
                'Failed to join two Dataframes')
            sys.exit(1)

    def normalizer(self) -> pd.DataFrame:
        """
        Returns dataframe with normalized columns
        """
        nrm = Normalizer()
        res = pd.DataFrame(
            nrm.fit_transform(
                self.df[self.get_numerical_columns(self.df)]), columns=self.get_numerical_columns(self.df)
        )
        return res

    def drop_unwanted_cols(self, columns: list) -> pd.DataFrame:
        """
        Drops columns which doesn't add to the model training
        ------------------
        columns:
            Type: list 
        Returns:
        ---------------
        pd.DataFrame
        """

        self.df.drop(columns=columns, axis=1, inplace=True)
        self.logger.info("Successfully droped unwanted columns")
        return self.df

    def remove_single_val_columns(self) -> pd.DataFrame:
        vc = self.df.nunique()
        col =[]
        for i in len(v):
            if list(vc)[i] ==1:
                col.append(v.index[i])
        self.df.drop(columns=col,axis=1,inplace = True)
        return self.df

    def change_columns_type_to(self, cols: list, data_type: str) -> pd.DataFrame:
        """
        Returns a DataFrame where the specified columns data types are changed to the specified data type
        Parameters
        ----------
        cols:
            Type: list
        data_type:
            Type: str
        Returns
        -------
        pd.DataFrame
        """
        try:
            for col in cols:
                self.df[col] = self.df[col].astype(data_type)
        except:
            print('Failed to change columns type')
        self.logger.info(f"Successfully changed columns type to {data_type}")
        return self.df

    def pipeline(self) -> pd.DataFrame:
        """
        performs a pipiline of cleaning methods in the given dataframe
        """

        return self.df
