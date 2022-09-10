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
        if (deep):
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
        self.df = self.df.drop_duplicates()

        return self.df

    def compute_holidays_gap(self, column):

        # Identify holiday dates and create dataframe of them
        holiday_rep = ['a', 'b', 'c']
        holidays = self.df[self.df.StateHoliday.isin(
            holiday_rep)].Date.unique()
        holidays = np.append(holidays, '2015-10-03')
        holidays.sort()

        #
        holidays = pd.to_datetime(holidays, dayfirst=True)
        df_temp = pd.DataFrame()
        days = self.df[column].unique()
        days.sort()
        df_temp['date'] = days
        df_temp['date'] = pd.to_datetime(df_temp.date, dayfirst=True)
        df_temp1 = pd.DataFrame({'date1': holidays})
        df_temp = pd.merge_asof(
            df_temp, df_temp1, left_on='date', right_on='date1', direction='forward')
        df_temp = pd.merge_asof(
            df_temp, df_temp1, left_on='date', right_on='date1')

        df_temp['Until_Holiday'] = df_temp.pop(
            'date1_x').sub(df_temp['date']).dt.days
        df_temp['Since_Holiday'] = df_temp['date'].sub(
            df_temp.pop('date1_y')).dt.days

        # df_temp.rename(columns={'date':'date'},inplace=True)

        self.df['date'] = pd.to_datetime(self.df.Date)
        self.df = self.df.merge(df_temp, on='date', how='inner')
        self.df.drop(columns=['date'], axis=1, inplace=True)

        return self.df

    def map_days(self):
        # Reshaping the time stamp representations
        self.df['Year'] = self.df['Date'].dt.year
        self.df['Month'] = self.df['Date'].dt.month
        self.df['Quarter'] = self.df['Date'].dt.quarter
        self.df['Week'] = self.df['Date'].dt.week
        self.df['Day'] = self.df['Date'].dt.day
        self.df['WeekOfYear'] = self.df['Date'].dt.weekofyear
        self.df['DayOfYear'] = self.df['Date'].dt.dayofyear
        self.df['IsWeekDay'] = (self.df.Date.dt.day_of_week < 5)*1

        # Converting since year and since week to months.
        self.df['CompetitionOpenMonthDuration'] = 12 * (self.df['Year'] - self.df['CompetitionOpenSinceYear']) + (
            self.df['Month'] - self.df['CompetitionOpenSinceMonth'])
        self.df['PromoOpenMonthDuration'] = 12 * (self.df['Year'] - self.df['Promo2SinceYear']) + (
            self.df['WeekOfYear'] - self.df['Promo2SinceWeek']) / 4.0

        self.df['Season'] = np.where(self.df['Month'].isin([3, 4, 5]), "Spring",
                                     np.where(self.df['Month'].isin([6, 7, 8]), "Summer",
                                              np.where(self.df['Month'].isin([9, 10, 11]), "Fall",
                                                       np.where(self.df['Month'].isin([12, 1, 2]), "Winter", "None"))))
        self.df['Month_Status'] = np.where(self.df['Day'].isin(np.arange(1, 11)), "Beginning",
                                           np.where(self.df['Day'].isin(np.arange(11, 21)), "Mid",
                                                    np.where(self.df['Day'].isin(np.arange(21, 32)), "End", "None")))
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
        col = []
        for i in len(v):
            if list(vc)[i] == 1:
                col.append(v.index[i])
        self.df.drop(columns=col, axis=1, inplace=True)
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

    def data_pipeline(self) -> pd.DataFrame:
        """
        performs a pipiline of cleaning methods in the given dataframe
        """
        unwanted_cols = ['Id','CompetitionOpenSinceYear','CompetitionOpenSinceMonth','Promo2SinceYear','Promo2SinceWeek']
        self.drop_null_entries();
        self.drop_duplicate();
        self.compute_holidays_gap('Date');
        self.convert_to_datetime('Date');
        self.map_days();
        self.drop_unwanted_cols(unwanted_cols);
        return self.df
