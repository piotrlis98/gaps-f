import calendar
import os
import shutil
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyimgur
import sklearn.neighbors._base
from sklearn import datasets
from sklearn import tree
from sklearn.impute import KNNImputer  # https://www.youtube.com/watch?v=m_qKhnaYZlc
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tqdm.auto import trange

sys.modules['sklearn.neighbors.base'] = sklearn.neighbors._base
from missingpy import MissForest

# CONFIG FOR END USER
if __name__ == '__main__':
    # chance = 75
    # reps = 100
    # imgurUpload = False

    chance = int(input('Wprowadź poziom utraty danych? [%]: '))
    reps = int(input('Wprowadź liczbę iteracji dla eksperymentu: '))
    imgurUpload = input('Umieścić rezultat na serwerze zewnętrznym? [T/N]: ')
    if imgurUpload == 'T':
        imgurUpload = True
    if imgurUpload == 'N':
        imgurUpload = False

    totalMethods = 5
    methods = ['zbiór oryginalny', 'średnia arytmetyczna', 'mediana', 'uzupełnianie naprzód', 'uzupełnianie wstecz',
               'interpolacja liniowa', 'KNN', 'MissForest']
    datasets_names = ['wine', 'iris', 'banknote', 'pima']

    # IMGUR CONFIG
    CLIENT_ID = 'eafe38dbbf1c2f6'
    im = pyimgur.Imgur(CLIENT_ID)
    # IMGUR CONFIG END

    # TIMESTAMP CONFIG

    ts = calendar.timegm(time.gmtime())
    path = 'temp/' + str(ts) + '_' + str(chance) + '%_' + 'x' + str(reps)
    isExist = os.path.exists(path)

    if not isExist:
        os.makedirs(path)
        print('\nUtworzono katalog: ', path)

    # TIMESTAMP CONFIG END

    chance /= 100


    #  http://ceur-ws.org/Vol-2136/10000108.pdf
    #  https://www.youtube.com/watch?v=EaGbS7eWSs0
    def input_nan(df, param, chance=chance):  # symulacja tracenia danych
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
        df_missing = input_nan(df, chance)
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
        df_missing = input_nan(df, 'Przypadki', chance)
        out_df = task_process(df_missing).round(0)
        out_df.to_csv('covid-gtm.csv', encoding='utf-8')
        out_df.to_excel('covid-gtm-czytelne.xlsx', encoding='utf-8')


    def currency_task():
        cols = [0, 3]
        names = ['Data', 'Kurs']
        df = pd.read_csv('D:\MEGA\Studia\Lisie pliki\Semestr 7\PD\wal.csv', parse_dates=True, usecols=cols,
                         names=names)
        df['Data'] = pd.to_datetime(df['Data']).dt.date
        df_missing = input_nan(df, 'Kurs', chance)
        out_df = task_process(df_missing).round(2)
        out_df.to_csv('kursy.csv', encoding='utf-8')
        out_df.to_excel('kursy-czytelne.xlsx', encoding='utf-8')


    def oil_task():
        cols = [0, 1]
        names = ['Data', 'Cena']
        df = pd.read_csv('D:\MEGA\Studia\Lisie pliki\Semestr 7\PD\oil.csv', parse_dates=True, usecols=cols,
                         names=names)
        df['Data'] = pd.to_datetime(df['Data']).dt.date
        df_missing = input_nan(df, 'Cena', chance)
        out_df = task_process(df_missing).round(2)
        out_df.to_csv('barylka.csv', encoding='utf-8')
        out_df.to_excel('barylka-czytelne.xlsx', encoding='utf-8')


    def births_task():
        cols = [1, 2]
        names = ['Data', 'Urodzenia']
        df = pd.read_csv('D:\MEGA\Studia\Lisie pliki\Semestr 7\PD\Births2015.csv', parse_dates=True, usecols=cols,
                         names=names)
        df['Data'] = pd.to_datetime(df['Data']).dt.date
        df_missing = input_nan(df, 'Urodzenia', chance)
        out_df = task_process(df_missing).round(0)
        out_df.to_csv('urodzeniaUS.csv', encoding='utf-8')
        out_df.to_excel('urodzeniaUS-czytelne.xlsx', encoding='utf-8')


    def handle_classify(x, y, names):
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.5)

        x_train_gaps = x_train.copy()

        for i in range(len(names)):
            x_train_gaps = input_nan(x_train_gaps, names[i], chance)

        x_train_mean = x_train_gaps.copy()
        x_train_median = x_train_gaps.copy()
        x_train_linear = x_train_gaps.copy()
        x_train_bfill = x_train_gaps.copy()
        x_train_ffill = x_train_gaps.copy()
        x_train_KNN = x_train_gaps.copy()
        x_train_MissForest = x_train_gaps.copy()

        imputer = KNNImputer(missing_values=np.nan)  # https://scikit-learn.org/stable/modules/impute.html
        imputer2 = MissForest()  # https://towardsdatascience.com/how-to-use-python-and-missforest-algorithm-to-impute-missing-data-ed45eb47cb9a

        x_train_mean_filled = x_train_mean.fillna(value=x_train_mean.mean(numeric_only=True)).round(1).iloc[:,
                              len(names):len(names) * 2]
        x_train_mean_filled.columns = x_train_mean_filled.columns.str.replace("Utracone", "średnia ar.")

        x_train_median_filled = x_train_median.fillna(value=x_train_median.median(numeric_only=True)).round(1).iloc[:,
                                len(names):len(names) * 2]
        x_train_median_filled.columns = x_train_median_filled.columns.str.replace("Utracone", "mediana")

        x_train_linear_filled = x_train_linear.interpolate(method='linear').bfill().ffill().round(1).iloc[:,
                                len(names):len(names) * 2]
        x_train_linear_filled.columns = x_train_linear_filled.columns.str.replace("Utracone", "interp")

        x_train_bfill_filled = x_train_bfill.fillna(method='bfill').round(1).iloc[:, len(names):len(names) * 2]
        x_train_bfill_filled.columns = x_train_bfill_filled.columns.str.replace("Utracone", "bfill")

        x_train_ffill_filled = x_train_ffill.fillna(method='ffill').round(1).iloc[:, len(names):len(names) * 2]
        x_train_ffill_filled.columns = x_train_ffill_filled.columns.str.replace("Utracone", "ffill")

        x_train_KNN_filled = pd.DataFrame(imputer.fit_transform(x_train_KNN), columns=x_train_KNN.columns).round(
            1).iloc[:,
                             len(names):len(names) * 2]
        x_train_KNN_filled.columns = x_train_KNN_filled.columns.str.replace("Utracone", "KNN")

        x_train_MissForest_filled = pd.DataFrame(imputer2.fit_transform(x_train_MissForest),
                                                 columns=x_train_MissForest.columns).round(1).iloc[:,
                                    len(names):len(names) * 2]
        x_train_MissForest_filled.columns = x_train_MissForest_filled.columns.str.replace("Utracone", "KNN")

        x_train_bfill_filled = x_train_bfill_filled.fillna(value=x_train_bfill_filled.mean(numeric_only=True))
        x_train_ffill_filled = x_train_ffill_filled.fillna(value=x_train_ffill_filled.mean(numeric_only=True))

        # print(x_train_bfill)

        classifier_default = tree.DecisionTreeClassifier()  # klasyfikator nauczony na oryginalnym zbiorze
        classifier_mean = tree.DecisionTreeClassifier()
        classifier_median = tree.DecisionTreeClassifier()
        classifier_bfill = tree.DecisionTreeClassifier()
        classifier_ffill = tree.DecisionTreeClassifier()
        classifier_linear = tree.DecisionTreeClassifier()
        classifier_KNN = tree.DecisionTreeClassifier()
        classifier_MissForest = tree.DecisionTreeClassifier()
        #  classifier = neighbors.KNeighborsClassifier()

        classifier_default.fit(x_train.values, y_train)
        classifier_mean.fit(x_train_mean_filled.values, y_train)
        classifier_median.fit(x_train_median_filled.values, y_train)
        classifier_bfill.fit(x_train_bfill_filled.values, y_train)
        classifier_ffill.fit(x_train_ffill_filled.values, y_train)
        classifier_linear.fit(x_train_linear_filled.values, y_train)
        classifier_KNN.fit(x_train_KNN_filled.values, y_train)
        classifier_MissForest.fit(x_train_MissForest_filled.values, y_train)

        predictions_default = classifier_default.predict(x_test.values)
        predictions_mean = classifier_mean.predict(x_test.values)
        predictions_ffill = classifier_ffill.predict(x_test.values)
        predictions_bfill = classifier_bfill.predict(x_test.values)
        predictions_linear = classifier_linear.predict(x_test.values)
        predictions_median = classifier_median.predict(x_test.values)
        predictions_KNN = classifier_KNN.predict(x_test.values)
        predictions_MissForest = classifier_MissForest.predict(x_test.values)

        acc_default = accuracy_score(y_test, predictions_default).round(4)
        acc_mean = accuracy_score(y_test, predictions_mean).round(4)
        acc_median = accuracy_score(y_test, predictions_median).round(4)
        acc_ffill = accuracy_score(y_test, predictions_ffill).round(4)
        acc_bfill = accuracy_score(y_test, predictions_bfill).round(4)
        acc_linear = accuracy_score(y_test, predictions_linear).round(4)
        acc_KNN = accuracy_score(y_test, predictions_KNN).round(4)
        acc_MissForest = accuracy_score(y_test, predictions_MissForest).round(4)

        return acc_default, acc_mean, acc_median, acc_ffill, acc_bfill, acc_linear, acc_KNN, acc_MissForest


    def iris_classify():
        iris = datasets.load_iris()

        x = pd.DataFrame(iris.data)
        y = pd.DataFrame(iris.target)

        names = ['długość kielicha', 'szerokość kielicha', 'długość płatka', 'szerokość płatka']
        x.columns = names

        return handle_classify(x, y, names)


    def wine_classify():
        wine = datasets.load_wine()

        x = pd.DataFrame(wine.data)
        y = pd.DataFrame(wine.target)

        names = ['alkohol', 'kwas jabłkowy', 'popiół', 'alkaliczność popiołu', 'magnez', 'fenole', 'flawanoidy',
                 'fenole nieflawanoidowe', 'proantocyjaniny', 'intensywność koloru', 'odcień', 'OD280/OD315', 'prolina']

        x.columns = names

        return handle_classify(x, y, names)


    # https://scikit-learn.org/stable/datasets/toy_dataset.html
    # https://www.activestate.com/resources/quick-reads/how-to-classify-data-in-python/
    # https://machinelearningmastery.com/standard-machine-learning-datasets/

    def banknote_classify():
        names = ['wariancja', 'skośność', 'kurtoza', 'entropia']
        data = pd.read_csv('http://lp44404.zut.edu.pl/datasets/data_banknote_authentication.txt', sep=",", header=None)

        x = data.iloc[:, 0:-1]
        y = data.iloc[:, -1:]
        x.columns = names

        return handle_classify(x, y, names)


    def pima_classify():  # https://www.datacamp.com/community/tutorials/decision-tree-classification-python
        names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age']
        data = pd.read_csv('http://lp44404.zut.edu.pl/datasets/pima-indians-diabetes.csv', sep=",", header=None)

        x = data.iloc[:, 0:-1]
        y = data.iloc[:, -1:]
        x.columns = names

        return handle_classify(x, y, names)


    res_banknote = []
    for i in trange(reps):
        res_banknote.append(banknote_classify())

    res_banknote = np.array(res_banknote)
    banknote_mean = res_banknote.mean(axis=0)
    print('\nBanknote -- ', banknote_mean)

    res_iris = []
    for i in trange(reps):
        res_iris.append(iris_classify())

    res_iris = np.array(res_iris)
    iris_mean = res_iris.mean(axis=0)
    print('\nIris -- ', iris_mean)

    res_wine = []
    for i in trange(reps):
        res_wine.append(wine_classify())

    res_wine = np.array(res_wine)
    wine_mean = res_wine.mean(axis=0)
    print('\nWine -- ', wine_mean)

    res_pima = []
    for i in trange(reps):
        res_pima.append(pima_classify())

    res_pima = np.array(res_pima)
    pima_mean = res_pima.mean(axis=0)
    print('\nPima -- ', pima_mean)


    def plot_comparsion(dataset_given, dataset_name, bar_color, toImgur=False):
        df = pd.DataFrame({'Dokładność': dataset_given * 100}, index=methods)

        # plot https://stackoverflow.com/questions/28931224/adding-value-labels-on-a-matplotlib-bar-chart
        ax = df.plot(kind='bar', figsize=(10, 8),
                     title='Zbiór danych: ' + dataset_name + ' | Iteracji: ' + str(
                         reps) + ' | Poziom utraty danych w zbiorze ' + dataset_name + ': ' + str(
                         round(chance * 100)) + "%",
                     xlabel='Metoda', ylabel='Dokładność [%]', legend=False, color=bar_color, zorder=3)

        # annotate
        for c in ax.containers:
            ax.bar_label(c, label_type='edge', padding=-12, color='white', fmt='%.2f%%')

        # pad the spacing between the number and the edge of the figure
        ax.margins(y=0.1)
        plt.ylim(50, 100)
        plt.axhline(y=dataset_given[0] * 100, linewidth=0.6, linestyle='--', alpha=0.5, color='k', zorder=1)
        plt.tight_layout()
        # plt.show()

        if toImgur:

            tempPath = str(path) + '/' + str(dataset_name) + '_' + str(chance * 100) + '%_' + str(reps) + '_' + str(
                ts) + '.png'
            plt.savefig(tempPath)
            upload_to_imgur = im.upload_image(tempPath,
                                              title=tempPath)
            print(upload_to_imgur.link, 'Tytuł: ', upload_to_imgur.title)

        else:
            tempPath = str(path) + '/' + str(dataset_name) + '_' + str(chance * 100) + '%_' + str(reps) + '.png'
            plt.savefig(tempPath)




    plot_comparsion(wine_mean, "wine", "#820064", toImgur=imgurUpload)
    plot_comparsion(iris_mean, "iris", "#0560e8", toImgur=imgurUpload)
    plot_comparsion(banknote_mean, "banknote", "#008519", toImgur=imgurUpload)
    plot_comparsion(pima_mean, 'pima', 'red', toImgur=imgurUpload)
    shutil.rmtree(path)
