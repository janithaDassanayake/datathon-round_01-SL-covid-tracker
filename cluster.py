import datetime
from datetime import datetime
from datetime import timedelta
import pandas as pd
pd.options.mode.chained_assignment = None
import numpy as np
from sklearn.linear_model import LinearRegression
previousreg = 1

regression_cluster_pred = {}

# function used to import dataset
def importCSV(path):
    load = pd.read_csv(path, encoding="ISO-8859-1")
    return load


def GetConfirm(n):
    global previousreg
    # get teh dummation
    previousreg = n + previousreg
    # return the output value
    return previousreg


def GetClusterName(n):
    clusternames = ['Divulapitiya', 'Welisara', 'Peliyagoda', 'Paliyagoda']
    notes = n.split()

    if ('Peliyagoda' in notes):
        return 'Paliyagoda'
    elif ('Paliyagoda' in notes):
        return 'Paliyagoda'
    elif ('Welisara' in notes):
        return 'Welisara'
    elif ('Divulapitiya' in notes):
        return 'Divulapitiya'
    else:
        return 'Other'


def PreprocessingCluster():
    # import the district details
    district_details = importCSV('clusterFile/cluster_data.csv').copy()
    # get the relavant columns
    filtercol = district_details[['Date Announced', 'Notes']]
    # remove null rows
    filtercol.dropna(inplace=True)
    # convert dates
    filtercol['Date'] = pd.to_datetime(filtercol['Date Announced'])
    filtercol.drop('Date Announced', axis='columns', inplace=True)
    filtercol = filtercol[filtercol['Date'] > '2020-10-01']
    # get the cluster names
    filtercol['cluster'] = [GetClusterName(c) for c in filtercol.Notes]

    return filtercol.reset_index()


def GetClusterPrediction():
    global previousreg
    global regression_cluster_pred
    clusterdf = PreprocessingCluster().copy()
    clusterwise = clusterdf.groupby(["cluster", "Date"])["Date"].count().reset_index(name="ConfirmPerday").copy()

    # loop each cluster and predict teh count
    clusternames = ['Divulapitiya', 'Paliyagoda']
    for clusname in clusternames:
        tempdf = clusterwise[clusterwise['cluster'] == clusname][['Date', 'ConfirmPerday']]
        # now sort the dates
        tempdf.sort_values(by=['Date'], inplace=True)
        tempdf.reset_index()
        tempdf.reset_index(drop=True, inplace=True)
        # iniialize the previosus which will need in the getconfirm function
        previousreg = tempdf['ConfirmPerday'][0]
        # calculate the previous total
        tempdf['Confirm'] = [GetConfirm(c) for c in tempdf.ConfirmPerday]
        tempdf.reset_index(drop=True, inplace=True)
        # initialize preious which is used for the count
        previousreg = tempdf['ConfirmPerday'][0]

        tempd = []
        tempt = []

        # get the each value to a 2d list
        for index, row in tempdf.iterrows():
            # temp get the date and y values to the anothe temp
            datesplit = row['Date'].strftime('%Y-%m-%d').split('-')
            tempd.append([int(datesplit[1]), int(datesplit[2])])
            tempt.append(int(row['ConfirmPerday']))

            # perform the regression analysis then

        regressionAnalysis(tempd, tempt, clusname, tempdf['Confirm'][len(tempdf) - 1])
    return regression_cluster_pred






def regressionAnalysis(x, y, cluster_name, finaldatetotal):
    global regression_cluster_pred
    X = np.array(x)
    y = y

    reg = LinearRegression().fit(X, y)
    reg.score(X, y)

    reg.coef_
    reg.intercept_
    createdate = '2020' + '-' + str(x[len(x) - 1][0]) + '-' + str(x[len(x) - 1][1])

    # add the upcoming two days
    adjustdate1 = datetime.strptime(createdate, '%Y-%m-%d') + timedelta(1)
    adjustdate2 = adjustdate1 + timedelta(1)

    # split the date
    datesplit = adjustdate1.strftime('%Y-%m-%d').split('-')
    pred1 = reg.predict(np.array([[int(datesplit[1]), int(datesplit[2])]]))[0]
    datesplit = adjustdate2.strftime('%Y-%m-%d').split('-')
    pred2 = reg.predict(np.array([[int(datesplit[1]), int(datesplit[2])]]))[0]
    regression_cluster_pred[cluster_name] = {'prediction_1': pred1 + finaldatetotal,
                                             'prediction_2': pred2 + pred1 + finaldatetotal,
                                             'date_1': adjustdate1.strftime('%Y-%m-%d'),
                                             'date_2': adjustdate2.strftime('%Y-%m-%d'),
                                             'increase_day_1': pred1,
                                             'increase_day_2': pred2

                                      }

