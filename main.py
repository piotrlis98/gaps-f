import numpy as np
from scipy import ndimage as nd
import pandas as pd
import random
import math as m
import matplotlib.pyplot as plt
import time
import openpyxl
from sklearn import datasets
import reportlab

totalMethods = 5


#  http://ceur-ws.org/Vol-2136/10000108.pdf
#  https://www.youtube.com/watch?v=EaGbS7eWSs0
def input_nan(df, param, chance=0.2):  # symulacja tracenia danych
    df_copy = df.copy()
    nan_percent = {param: chance}  # szansa na wystąpienie utraconej wartości
    for col, perc in nan_percent.items():
        df_copy['null'] = np.random.choice([0, 1], size=df_copy.shape[0], p=[1 - perc, perc])
        df_copy.loc[df_copy['null'] == 1, col] = np.nan
    df_copy.drop(columns=['null'], inplace=True)
    df[param + '(Utracone)'] = df_copy[param]
    return df


def fill_with_mean(df):  # uzupełnienie brakujących danych na podstawie średniej wartości
    dataframe = df.copy()
    dataframe = dataframe.fillna(value=df.mean(numeric_only=True))
    df['Średnia arytmetyczna'] = dataframe.iloc[:, -1]
    return df


def fill_with_median(df):  # uzupełnienie brakujących danych na podstawie mediany
    dataframe = df.copy()
    dataframe = dataframe.fillna(value=df.median(numeric_only=True))
    df['Mediana'] = dataframe.iloc[:, -2]
    return df


def forward_fill(df):  # uzupełnianie 'do przodu'
    dataframe = df.copy()
    dataframe = dataframe.fillna(method='ffill')
    df['Naprzód'] = dataframe.iloc[:, -3]
    return df


def backward_fill(df):  # uzupełnianie 'wstecz'
    dataframe = df.copy()
    dataframe = dataframe.fillna(method='bfill')
    df['Wstecz'] = dataframe.iloc[:, -4]
    return df


#  https://pandas.pydata.org/docs/reference/api/pandas.Series.interpolate.html
def fill_by_interpolation(df, type='linear'):  # interpolacja
    dataframe = df.copy()
    dataframe = dataframe.interpolate(method=type).bfill().ffill()
    df['Interpolacja'] = dataframe.iloc[:, -5]
    return df


def task_process(df, interp_method='linear'):
    dataframe = df.copy()
    df1 = fill_with_mean(dataframe)
    df2 = fill_with_median(df1)
    df3 = forward_fill(df2)
    df4 = backward_fill(df3)
    df5 = fill_by_interpolation(df4, interp_method)
    return df5


def temperatures_task():
    cols = [1, 2, 3]
    names = ['Miasto', 'Data', 'Temperatura']
    df = pd.read_csv('D:\MEGA\Studia\Lisie pliki\Semestr 7\PD\Szczecin2020.csv', parse_dates=True, usecols=cols,
                     names=names)
    df['Data'] = pd.to_datetime(df['Data']).dt.date
    df_missing = input_nan(df, 0.2)
    out_df = task_process(df_missing).round(1)
    out_df.to_csv('temperatury.csv', encoding='utf-8')
    out_df.to_excel('temperatury_czytelne.xlsx', encoding='utf-8')


#  std:  https://stackoverflow.com/questions/57748856/standard-deviation-for-the-difference-of-two-dataframes-with-group-by


def guatemala_covid_task():
    cols = [2, 3, 5]
    names = ['Kraj', 'Data', 'Przypadki']
    df = pd.read_csv('D:\MEGA\Studia\Lisie pliki\Semestr 7\PD\guatemala-covid.csv', parse_dates=True, usecols=cols,
                     names=names)
    df['Data'] = pd.to_datetime(df['Data']).dt.date
    df_missing = input_nan(df, 'Przypadki', 0.2)
    out_df = task_process(df_missing).round(0)
    out_df.to_csv('covid-gtm.csv', encoding='utf-8')
    out_df.to_excel('covid-gtm-czytelne.xlsx', encoding='utf-8')


def currency_task():
    cols = [0, 3]
    names = ['Data', 'Kurs']
    df = pd.read_csv('D:\MEGA\Studia\Lisie pliki\Semestr 7\PD\wal.csv', parse_dates=True, usecols=cols,
                     names=names)
    df['Data'] = pd.to_datetime(df['Data']).dt.date
    df_missing = input_nan(df, 'Kurs', 0.2)
    out_df = task_process(df_missing).round(2)
    out_df.to_csv('kursy.csv', encoding='utf-8')
    out_df.to_excel('kursy-czytelne.xlsx', encoding='utf-8')


def oil_task():
    cols = [0, 1]
    names = ['Data', 'Cena']
    df = pd.read_csv('D:\MEGA\Studia\Lisie pliki\Semestr 7\PD\oil.csv', parse_dates=True, usecols=cols,
                     names=names)
    df['Data'] = pd.to_datetime(df['Data']).dt.date
    df_missing = input_nan(df, 'Cena', 0.2)
    out_df = task_process(df_missing).round(2)
    out_df.to_csv('barylka.csv', encoding='utf-8')
    out_df.to_excel('barylka-czytelne.xlsx', encoding='utf-8')


def births_task():
    cols = [1, 2]
    names = ['Data', 'Urodzenia']
    df = pd.read_csv('D:\MEGA\Studia\Lisie pliki\Semestr 7\PD\Births2015.csv', parse_dates=True, usecols=cols,
                     names=names)
    df['Data'] = pd.to_datetime(df['Data']).dt.date
    df_missing = input_nan(df, 'Urodzenia', 0.2)
    out_df = task_process(df_missing).round(0)
    out_df.to_csv('urodzeniaUS.csv', encoding='utf-8')
    out_df.to_excel('urodzeniaUS-czytelne.xlsx', encoding='utf-8')


def iris_task():  # https://archive.ics.uci.edu/ml/datasets/iris
    iris = datasets.load_iris()
    X = iris.data[:, :4]
    Y = iris.target
    names = ['długość kielicha', 'szerokość kielicha', 'długość płatka', 'szerokość płatka']
    df = pd.DataFrame(data=X, columns=names)
    df_missing = df.copy()

    for i in range(len(names)):
        df_missing = input_nan(df, names[i], 0.2)

    df_missing['target'] = Y
    df_missing.loc[df_missing['target'] == 0, 'nazwa'] = 'Iris Setosa'
    df_missing.loc[df_missing['target'] == 1, 'nazwa'] = 'Iris Versicolour'
    df_missing.loc[df_missing['target'] == 2, 'nazwa'] = 'Iris Virginica'
    # df_missing = df_missing.drop(['target'], axis=1)

    # df_missing.to_excel('iris_test.xlsx', encoding='utf-8')

    df_fill = df_missing.copy()

    df_new = df_fill.fillna(value=df.mean(numeric_only=True))
    df_new.columns = df_new.columns.str.replace("Utracone", "Średnia ar.")
    df_out = pd.concat([df_missing, df_new.iloc[:, 4:8]], axis=1)
    df_out.to_excel('iristask/iris_mean_test.xlsx', encoding='utf-8')

    df_new = df_fill.fillna(value=df.median(numeric_only=True))
    df_new.columns = df_new.columns.str.replace("Utracone", "mediana")
    df_out = pd.concat([df_missing, df_new.iloc[:, 4:8]], axis=1)
    df_out.to_excel('iristask/iris_median_test.xlsx', encoding='utf-8')

    df_new = df_fill.fillna(method='ffill')
    df_new.columns = df_new.columns.str.replace("Utracone", "naprzód")
    df_out = pd.concat([df_missing, df_new.iloc[:, 4:8]], axis=1)
    df_out.to_excel('iristask/iris_ffill_test.xlsx', encoding='utf-8')

    df_new = df_fill.fillna(method='bfill')
    df_new.columns = df_new.columns.str.replace("Utracone", "wstecz")
    df_out = pd.concat([df_missing, df_new.iloc[:, 4:8]], axis=1)
    df_out.to_excel('iristask/iris_bfill_test.xlsx', encoding='utf-8')

    df_new = df_fill.interpolate(method='linear').bfill().ffill()
    df_new.columns = df_new.columns.str.replace("Utracone", "interp. lin.")
    df_out = pd.concat([df_missing, df_new.iloc[:, 4:8]], axis=1).round(1)
    df_out.to_excel('iristask/iris_linear_test.xlsx', encoding='utf-8')


def wine_task():  # https://archive.ics.uci.edu/ml/datasets/wine
    wine = datasets.load_wine()
    X = wine.data[:, :13]
    Y = wine.target
    names = ['alkohol', 'kwas jabłkowy', 'popiół', 'alkaliczność popiołu', 'magnez', 'fenole', 'flawanoidy',
             'fenole nieflawanoidowe', 'proantocyjaniny', 'intensywność koloru', 'odcień', 'OD280/OD315', 'prolina']
    df = pd.DataFrame(data=X, columns=names)

    df_missing = df.copy()
    for i in range(len(names)):
        df_missing = input_nan(df, names[i], 0.2)

    df_missing['target'] = Y

    df_fill = df_missing.copy()

    df_new = df_fill.fillna(value=df.mean(numeric_only=True))
    df_new.columns = df_new.columns.str.replace("Utracone", "Średnia ar.")
    df_out = pd.concat([df_missing, df_new.iloc[:, 13:26]], axis=1)
    df_out.to_excel('winetask/iris_mean_test.xlsx', encoding='utf-8')

    df_new = df_fill.fillna(value=df.median(numeric_only=True))
    df_new.columns = df_new.columns.str.replace("Utracone", "mediana")
    df_out = pd.concat([df_missing, df_new.iloc[:, 13:26]], axis=1)
    df_out.to_excel('winetask/iris_median_test.xlsx', encoding='utf-8')

    df_new = df_fill.fillna(method='ffill')
    df_new.columns = df_new.columns.str.replace("Utracone", "naprzód")
    df_out = pd.concat([df_missing, df_new.iloc[:, 13:26]], axis=1)
    df_out.to_excel('winetask/iris_ffill_test.xlsx', encoding='utf-8')

    df_new = df_fill.fillna(method='bfill')
    df_new.columns = df_new.columns.str.replace("Utracone", "wstecz")
    df_out = pd.concat([df_missing, df_new.iloc[:, 13:26]], axis=1)
    df_out.to_excel('winetask/iris_bfill_test.xlsx', encoding='utf-8')

    df_new = df_fill.interpolate(method='linear').bfill().ffill()
    df_new.columns = df_new.columns.str.replace("Utracone", "interp. lin.")
    df_out = pd.concat([df_missing, df_new.iloc[:, 13:26]], axis=1).round(1)
    df_out.to_excel('winetask/iris_linear_test.xlsx', encoding='utf-8')


wine_task()
