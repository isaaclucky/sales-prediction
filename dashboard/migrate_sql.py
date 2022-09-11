import os
import pandas as pd
import mysql.connector as mysql
from mysql.connector import Error


def DBConnect(dbName=None):

    conn = mysql.connect(host='localhost', user='owon', password='lucky1Z_',
                         database=dbName, buffered=True)
    cur = conn.cursor()
    return conn, cur


def createDB(dbName: str) -> None:

    conn, cur = DBConnect()
    cur.execute(f"CREATE DATABASE IF NOT EXISTS {dbName};")
    conn.commit()
    cur.close()


def createTables(dbName: str) -> None:

    conn, cur = DBConnect(dbName)
    sqlFile = 'dashboard/schema.sql'
    fd = open(sqlFile, 'r')
    readSqlFile = fd.read()
    fd.close()

    sqlCommands = readSqlFile.split(';')

    for command in sqlCommands:
        try:
            res = cur.execute(command)
        except Exception as ex:
            print("Command skipped: ", command)
            print(ex)
    conn.commit()
    cur.close()

    return


def insert_to_table(dbName: str, df: pd.DataFrame, table_name: str) -> None:

    conn, cur = DBConnect(dbName)

    for _, row in df.iterrows():
        sqlQuery = f"""INSERT INTO {table_name}( 'Store',
        'DayOfWeek',
        'Sales',
       'Open', 
       'Promo', 
       'StateHoliday',
       'SchoolHoliday', 
       'StoreType',
       'Assortment', 
       'CompetitionDistance',
       'Promo2', 
       'PromoInterval',
       'Until_Holiday', 
       'Since_Holiday', 
       'Year', 
       'Month', 
       'Quarter',
       'Week',
       'Day',
       'WeekOfYear',
       'DayOfYear',
       'IsWeekDay',
       'CompetitionOpenMonthDuration', 
       'PromoOpenMonthDuration',
       'Season',
       'Month_Status')  
        VALUES(%s,%s,%s,%s,%s,%s,%s,%s
                    %s,%s,%s,%s,%s,%s,%s,%s
                    %s,%s,%s,%s,%s,%s,%s,%s,%s,%s
                    )"""
        data = (row[0], row[1], row[2], row[3], row[4], row[5],
                row[6], row[7], row[8], row[9], row[10], row[11], row[12], row[13],
                row[14], row[15], row[16], row[17], row[18], row[19], row[20],
                row[21], row[22], row[23], row[24], row[25])

        try:
            # Execute the SQL command
            cur.execute(sqlQuery, data)
            # Commit your changes in the database
            conn.commit()
            print("Data Inserted Successfully")
        except Exception as e:
            conn.rollback()
            print("Error: ", e)
    return


def db_execute_fetch(*args, many=False, tablename='', rdf=True, **kwargs) -> pd.DataFrame:

    connection, cursor1 = DBConnect(**kwargs)
    if many:
        cursor1.executemany(*args)
    else:
        cursor1.execute(*args)

    # get column names
    field_names = [i[0] for i in cursor1.description]

    # get column values
    res = cursor1.fetchall()

    # get row count and show info
    nrow = cursor1.rowcount
    if tablename:
        print(f"{nrow} recrods fetched from {tablename} table")

    cursor1.close()
    connection.close()

    # return result
    if rdf:
        return pd.DataFrame(res, columns=field_names)
    else:
        return res


if __name__ == "__main__":

    createTables(dbName='SalesPrediction')
    df = pd.read_csv('data/df_train_prep.csv')
    df = df[:100]
    df.drop(['Unnamed: 0', 'Date','Customers'], axis=1, inplace=True)
    insert_to_table(dbName='SalesPrediction', df=df,
                    table_name='                                                                                                                                                                                                                                        ')
