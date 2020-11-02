import matplotlib
import pandas as pd
import numpy as np
import utils as utils
import matplotlib.pyplot as plt
import get_covid_updated_data as sl_data
from pathlib import Path
import os
from tqdm.notebook import tqdm
from scipy.integrate import solve_ivp
import numpy
import datetime
from datetime import timedelta
from sklearn.model_selection import train_test_split
import math
import calendar
import os
from datetime import datetime


pd.options.mode.chained_assignment = None  # default='warn'
# %matplotlib inline

# initialize global variables
# initialize global variables
loaddf = pd.DataFrame()
train = pd.DataFrame()
train_loc = pd.DataFrame()
# used in the latter part calculations
previous = ''
max_days = 0

# varibale needed for the SEIR calculaions
count = -1
result = 1
# to store current values
currentconfirmed = []

# to store day number for the district issues
days = []

# to store the district population dataframe
district_details = pd.DataFrame()

# to store predicted values
predictedvaluesSEIR_x = []
predictedvaluesSEIR_y = []

# to store the past data
valuesSEIR_x = []
valuesSEIR_y = []

# to store the prediction of exposure,death, infection
rec_cordinate_y = []
inf_cordinate_y = []
exp_cordinate_y = []

# to store the populations
N = 0

# to store each district and csv
all_district_info = {}
all_district_details = {}

reproduction_number = {'Colombo': 2.6,
                       'Anuradhapura': 1 / 5,
                       'Badulla': 13,
                       'Galle': 1.5,
                       'Gampaha': 2.2,
                       'Hambantota': 6,
                       'Jaffna': 1.6,
                       'Kalutara': 1.1,
                       'Kandy': 1 / 8,
                       'Kegalle': 1 / 6,
                       'Kurunegala': 1 / 3,
                       'Mannar': 7,
                       'Matale': 1 / 5.9,
                       'Matara': 1,
                       'Moneragala': 1 / 19,
                       'NuwaraEliya': 3.8,
                       'Puttalam': 2.4,
                       'Ratnapura': 2.6,
                       'Trincomalee': 1,
                       'Vavuniya': 1,
                       'Unspecified': 1.06}

# to save curennt district
district_current = ''


def get_data_for_SEIR():
    sl_all_covid_data = sl_data.get_all_sl_covid_data()
    df = sl_all_covid_data
    return df


def GetConfirmPerDay(n):
    global previous
    # get teh difference
    diff = n - previous
    # make the previous as current value
    previous = n
    # return the output value
    return diff


# when you gave the required datframe this perform teh necessary preprocessing
def preprocessing(df):
    global previous

    # first get the first row confirmed count
    previous = loaddf['Confirmed'][0]

    # create the confirmperday column whch wil be sued in the SEIR calculatios
    df['ConfirmPerday'] = [GetConfirmPerDay(c) for c in df.Confirmed]
    df['ConfirmPerday'][0] = 1.0

    df.reset_index(drop=True, inplace=True)

    # remove null rows
    df.dropna(inplace=True)

    # convert to date values
    df['Date'] = pd.to_datetime(df.Date)

    # get the second wave data
    df = df[df['Date'] > '2020-10-01']

    # now sort the dates
    df.sort_values(by=['Date'], inplace=True)

    df.index = range(len(df.index))

    return df


def TrainIntializing():
    global train
    global loaddf
    train = loaddf.copy()
    train = preprocessing(train).copy()


# Susceptible equation
def dS_dt(S, I, R_t, T_inf):
    return -(R_t / T_inf) * I * S


# Exposed equation
def dE_dt(S, E, I, R_t, T_inf, T_inc):
    return (R_t / T_inf) * I * S - (T_inc ** -1) * E


# Infected equation
def dI_dt(I, E, T_inc, T_inf):
    return (T_inc ** -1) * E - (T_inf ** -1) * I


# Recovered/Remove/deceased equation
def dR_dt(I, T_inf):
    return (T_inf ** -1) * I


def SEIR_model(t, y, R_t, T_inf, T_inc):
    if callable(R_t):
        reproduction = R_t(t)
    else:
        reproduction = R_t

    S, E, I, R = y

    S_out = dS_dt(S, I, reproduction, T_inf)
    E_out = dE_dt(S, E, I, reproduction, T_inf, T_inc)
    I_out = dI_dt(I, E, T_inc, T_inf)
    R_out = dR_dt(I, T_inf)

    return [S_out, E_out, I_out, R_out]


def TuneDataset():
    global loaddf
    global train
    global train_loc
    # import teh dataset
    loaddf = get_data_for_SEIR().copy()
    # df = covid_files[str(file_name)]
    # split the dataset
    TrainIntializing()
    train_loc = train.copy()

    return train_loc


# specially designed method for the all the district info fetching
def AllDistricts_TuneDataset(pass_dataframe):
    global loaddf
    global train
    global train_loc

    # import teh dataset
    loaddf = pass_dataframe.copy()

    # split the dataset
    TrainIntializing()

    train_loc = train.copy()

    return train_loc


def time_varying_reproduction(t):
    global count
    global max_days
    global result
    global train_loc
    global N

    count += 1

    if (count < max_days):
        num = - math.log((N - train_loc['ConfirmPerday'][count]) / (train_loc['Confirmed'][count]))
        den = ((train_loc['ConfirmPerday'][count] / N) - (1 - (train_loc['Confirmed'][count] / N)))
        result = (num / den) / 4.8
        currentconfirmed[count] = train_loc['Confirmed'][count]

        return result
    else:

        return result


def CountrySEIR(SLpopulation):
    global train_loc
    global max_days
    global currentconfirmed
    global count
    global N
    # get infected people intial
    n_infected = train_loc['Confirmed'].iloc[0]

    count = -1

    # sri lankan population
    N = SLpopulation

    max_days = len(train_loc)

    # Initial stat for SEIR model
    s = (N - n_infected) / N
    e = 0.
    i = n_infected / N
    r = 0.

    # Define all variable of SEIR model
    T_inc = 11.2  # average incubation period
    T_inf = 12  # average infectious period
    R_0 = 1.7  # reproduction number

    # to store current confirm values
    currentconfirmed = [0 for i in range(max_days + 2)]

    ## Solve the SEIR model
    sol = solve_ivp(SEIR_model, [0, max_days + 2], [s, e, i, r], args=(time_varying_reproduction, T_inf, T_inc),
                    t_eval=np.arange(max_days + 2))
    ## Plot result
    plot_model_and_predict(train_loc, N, sol, title='SEIR Model (with intervention)')


def plot_model_and_predict(data, pop, solution, title='SEIR model'):
    global predictedvaluesSEIR_x
    global predictedvaluesSEIR_y
    global valuesSEIR_x
    global valuesSEIR_y
    global district_current
    global exp_cordinate_y
    global inf_cordinate_y
    global rec_cordinate_y

    sus, exp, inf, rec = solution.y
    preds = np.clip((inf + rec) * pop, 0, np.inf)

    # get the predicted value to a array
    predictedvaluesSEIR_y = preds
    predictedvaluesSEIR_x = list(range(len(data) + 2))
    valuesSEIR_y = currentconfirmed

    valuesSEIR_x = list(range(len(data) + 2))
    rec_cordinate_y = list(rec)
    exp_cordinate_y = list(exp)
    inf_cordinate_y = list(inf)


'''Model predicted data plot '''


def FetchCordinates():
    global predictedvaluesSEIR_x
    global predictedvaluesSEIR_y
    global valuesSEIR_x
    global valuesSEIR_y

    return predictedvaluesSEIR_x, list(predictedvaluesSEIR_y), valuesSEIR_x, list(valuesSEIR_y)


# predictedvaluesSEIR_x = predicted x
# list(predictedvaluesSEIR_y) = predicted y
# valuesSEIR_x = current X
# list(valuesSEIR_y) = current y

'''===========================================for EIR Plot ============================================='''


def FetchCordinatedEIR():
    global exp_cordinate_y
    global inf_cordinate_y
    global rec_cordinate_y
    global valuesSEIR_x

    return exp_cordinate_y, inf_cordinate_y, rec_cordinate_y, valuesSEIR_x


# valuesSEIR_x = days
TuneDataset()
CountrySEIR(21000000)

''' ========================================================== '''


def importCSV(path):
    # read the csv
    load = pd.read_csv(path, encoding="ISO-8859-1")

    return load


# this is the method you need to call when predicting  district
# place the district_details csv
def PreprocessDistrictDetails(districtname):
    global district_details

    # import the district details
    district_details = importCSV('population_files/district_details.csv').copy()

    # get the required district population
    population = list(district_details.loc[district_details['ï»¿District'] == districtname]['Population'])[0]

    # there are commas in the numbers remove theme
    popu = population.split(',')

    return int(''.join(popu))


def time_varying_reproduction_District(t):
    global count
    global max_days
    global result
    global train_loc
    global N
    global reproduction_number
    global district_current

    count += 1

    if (count < max_days):
        num = - math.log((N - train_loc['ConfirmPerday'][count]) / (train_loc['Confirmed'][count]))
        den = ((train_loc['ConfirmPerday'][count] / N) - (1 - (train_loc['Confirmed'][count] / N)))
        result = (num / den) * (reproduction_number[district_current])
        currentconfirmed[count] = train_loc['Confirmed'][count]
        return result
    else:

        return result


def DistrictSEIR(districtname):
    global train_loc
    global max_days
    global currentconfirmed
    global count
    global N
    # get infected people intial
    n_infected = train_loc['Confirmed'].iloc[0]

    count = -1

    # sri lankan population
    if (districtname != 'Unspecified'):
        N = PreprocessDistrictDetails(districtname)
    else:
        N = 21000000

    max_days = len(train_loc)
    print(max_days)
    # Initial stat for SEIR model
    s = (N - n_infected) / N
    e = 0.
    i = n_infected / N
    r = 0.

    # Define all variable of SEIR model
    T_inc = 11.2  # average incubation period
    T_inf = 12  # average infectious period
    R_0 = 1.7  # reproduction number

    # to store current confirm values
    currentconfirmed = [0 for i in range(max_days + 2)]

    ## Solve the SEIR model
    sol = solve_ivp(SEIR_model, [0, max_days + 2], [s, e, i, r],
                    args=(time_varying_reproduction_District, T_inf, T_inc),
                    t_eval=np.arange(max_days + 2))
    ## Plot result
    model_and_predict_District(train_loc, N, sol, title='SEIR Model (with intervention)')


def model_and_predict_District(data, pop, solution, title='SEIR model'):
    global predictedvaluesSEIR_x
    global predictedvaluesSEIR_y
    global valuesSEIR_x
    global valuesSEIR_y
    global district_current
    global exp_cordinate_y
    global inf_cordinate_y
    global rec_cordinate_y

    sus, exp, inf, rec = solution.y
    preds = np.clip((inf + rec) * pop, 0, np.inf)
    # get the predicted value to a array

    predictedvaluesSEIR_y = preds
    predictedvaluesSEIR_x = list(range(train_loc['day'][0], train_loc['day'][0] + len(data) + 2))

    valuesSEIR_y = currentconfirmed
    valuesSEIR_x = list(range(train_loc['day'][0], train_loc['day'][0] + len(data) + 2))

    rec_cordinate_y = list(rec)
    exp_cordinate_y = list(exp)
    inf_cordinate_y = list(inf)


# read the csv


# read each file and save it to a dictionary
# {distrcitname : dataframe}
def AdjustDates():
    global valuesSEIR_x
    global predictedvaluesSEIR_x

    global train_loc

    valuesSEIR_x = []
    valuesSEIR_x = [d.strftime('%Y-%m-%d') for d in train_loc['Date'].tolist()]

    adjustdate1 = datetime.strptime(valuesSEIR_x[len(valuesSEIR_x) - 1], '%Y-%m-%d') + timedelta(1)
    adjustdate2 = adjustdate1 + timedelta(1)
    valuesSEIR_x.append(adjustdate1.strftime('%Y-%m-%d'))
    valuesSEIR_x.append(adjustdate2.strftime('%Y-%m-%d'))
    predictedvaluesSEIR_x = []
    predictedvaluesSEIR_x = valuesSEIR_x.copy()





def getTermSent():
    global all_district_info
    # get current working directory
    cd = os.getcwd()
    # path to the machine folder
    path = 'sl_covid_District_wise'
    filelist = os.listdir(path)

    # read the file through for loop
    for file in filelist:
        with open('sl_covid_District_wise/' + file, 'r') as f:
            # read the file and save it to a datframe and put it to a dictionary
            # import the district details
            district_read = importCSV(f.name).copy()

            splitterm = f.name.split('/')
            docname = splitterm[len(splitterm) - 1][:file.find(".csv")]

            # add to the dictionary
            all_district_info[docname] = district_read.copy()

    return all_district_info


# now read the predcited value of each district

# now read the predcited value of each district

def getAllDistrictInfo():
    count = 0
    global all_district_info
    global district_current
    global all_district_details
    # update the
    getTermSent()
    for district, datafrm in all_district_info.items():
        # run each district
        count += 1
        district_current = district
        AllDistricts_TuneDataset(datafrm)
        DistrictSEIR(district)
        # get the preicted patients cordinates
        predi_cordinates = FetchCordinates()[1]
        last_two_cordinates = predi_cordinates[len(predi_cordinates) - 2:]

        AdjustDates()
        all_district_details[district] = {
            'predict_x': FetchCordinates()[0].copy(),
            'predict_y': FetchCordinates()[1].copy(),
            'current_x': FetchCordinates()[0].copy(),
            'current_y': FetchCordinates()[3].copy(),
            'predicted_two_days': last_two_cordinates,
            'exp_cordinates_y': FetchCordinatedEIR()[0].copy(),
            'inf_cordinate_y': FetchCordinatedEIR()[1].copy(),
            'rec_cordinate_y': FetchCordinatedEIR()[2].copy(),
            'eir_cordinate_x': FetchCordinates()[0].copy()
        }

    return all_district_details