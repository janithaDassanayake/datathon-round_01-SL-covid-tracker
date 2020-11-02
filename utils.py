from sklearn.metrics import mean_squared_error
import numpy
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split

numpy.random.seed(10)


def create_covid_dataset(sl_dataset, val=1):
    covid_dataX, covid_dataY = [], []
    for i in range(len(sl_dataset) - val):
        x = sl_dataset[i:(i + val), 0]
        covid_dataX.append(x)
        covid_dataY.append(sl_dataset[i + val, 0])
    return numpy.array(covid_dataX), numpy.array(covid_dataY)


def create_sl_covid_preprocessed_Dataset(covid_df):
    covid_df.drop(covid_df.columns.difference(['Date', 'Confirmed', 'Recovered']), 1, inplace=True)
    covid_dfnew = covid_df
    covid_df = covid_df['Confirmed']
    covid_df1 = covid_dfnew['Recovered']

    covid_dataset = covid_df.values
    covid_dataset1 = covid_df1.values

    covid_dataset = covid_dataset.reshape(-1, 1)
    covid_dataset1 = covid_dataset1.reshape(-1, 1)

    covid_dataset = covid_dataset.astype('float32')
    covid_dataset1 = covid_dataset1.astype('float32')

    sl_train_size = len(covid_dataset) - 2
    sl_train, sl_test = covid_dataset[0:sl_train_size, :], covid_dataset[sl_train_size:len(covid_dataset), :]

    sl_train_size1 = len(covid_dataset1) - 2
    sl_train1, sl_test1 = covid_dataset1[0:sl_train_size1, :], covid_dataset1[sl_train_size1:len(covid_dataset1), :]

    val = 1
    sl_trainX, sl_trainY = create_covid_dataset(covid_dataset, val)
    sl_testX, sl_testY = create_covid_dataset(sl_test, val)

    sl_trainX_close, sl_trainY_close = create_covid_dataset(covid_dataset1, val)
    sl_testX_close, sl_testY_close = create_covid_dataset(sl_test1, val)

    return sl_trainX, sl_trainY, sl_testX, sl_testY, sl_trainX_close, sl_trainY_close, sl_testX_close, sl_testY_close


def get_sl_covid_Data(covid_df):
    dates = []
    sl_Confirmed = []
    sl_Recovered = []

    last_data_row = covid_df.tail(1)
    covid_df = covid_df.head(len(covid_df))
    df_dates = covid_df.loc[:, 'Date']
    df_Confirmed = covid_df.loc[:, 'Confirmed']
    df_Recovered = covid_df.loc[:, 'Recovered']

    for sl_date in df_dates:
        dates.append([int(sl_date.split('-')[2])])

    for Confirmed in df_Confirmed:
        sl_Confirmed.append(int(Confirmed))

    for Recovered in df_Recovered:
        sl_Recovered.append(int(Recovered))

    l_date = int(((list(last_data_row['Date']))[0]).split('-')[2])
    l_Confirmed = int((list(last_data_row['Confirmed']))[0])
    l_Recovered = int((list(last_data_row['Recovered']))[0])

    return dates, sl_Confirmed, l_date, l_Confirmed, sl_Recovered, l_Recovered


'''
=============================================================================================
                                 linear regression
'''


def linear_regression(dates, sl_data, sl_test_date, sl_df, forcastingDays):
    linear_reg = LinearRegression()
    sl_trainX, sl_trainY, sl_testX, sl_testY, sl_trainX_Recovered, sl_trainY_Recovered, sl_testX_Recovered, sl_testY_Recovered = create_sl_covid_preprocessed_Dataset(
        sl_df)

    X_train, X_test, y_train, y_test = train_test_split(sl_trainX, sl_trainY, test_size=0.2, random_state=20)
    X_train_Recovered, X_test_Recovered, y_train_Recovered, y_test_Recovered = train_test_split(sl_trainX_Recovered,
                                                                                                sl_trainY_Recovered,
                                                                                                test_size=0.2,
                                                                                                random_state=20)

    linear_reg.fit(sl_trainX, sl_trainY)
    linear_reg.fit(sl_trainX_Recovered, sl_trainY_Recovered)

    predict_decision_boundary = linear_reg.predict(sl_trainX)
    predict_decision_boundary_Recovered = linear_reg.predict(sl_trainX_Recovered)

    linear_reg_y_pred = linear_reg.predict(X_test)
    linear_reg_y_pred_Recovered = linear_reg.predict(X_test_Recovered)

    mean_squared_error_test_score = mean_squared_error(y_test, linear_reg_y_pred)
    mean_squared_error_test_score_Recovered = mean_squared_error(y_test_Recovered, linear_reg_y_pred_Recovered)

    prediction_of_linear_reg = linear_reg.predict(sl_testX)[0]
    prediction_of_linear_reg_Recovered = linear_reg.predict(sl_testX_Recovered)[0]

    recoveryList = predict_decision_boundary.tolist()

    df = pd.DataFrame(recoveryList)
    forecast_out = int(forcastingDays)

    df['Prediction'] = df.shift(-forecast_out)

    X = np.array(df.drop(['Prediction'], 1))
    X = preprocessing.scale(X)
    X_forecast = X[-forecast_out:]
    X = X[:-forecast_out]
    y = np.array(df['Prediction'])
    y = y[:-forecast_out]

    X_train_f, X_test_f, y_train_f, y_test_f = train_test_split(X, y, test_size=0.2)
    linear_reg.fit(X_train_f, y_train_f)
    accuracy = linear_reg.score(X_test_f, y_test_f)

    forecast_prediction = linear_reg.predict(X_forecast)

    return predict_decision_boundary, prediction_of_linear_reg, accuracy, prediction_of_linear_reg_Recovered, forecast_prediction, predict_decision_boundary_Recovered
