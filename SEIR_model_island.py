import pandas as pd
import numpy as np
import get_covid_updated_data as sl_data
from scipy.integrate import solve_ivp
import datetime
from datetime import timedelta
import math
from datetime import datetime
pd.options.mode.chained_assignment = None

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
        result = (num / den) / 4.75
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


def FetchCordinatedEIR():
    global exp_cordinate_y
    global inf_cordinate_y
    global rec_cordinate_y
    global valuesSEIR_x

    return exp_cordinate_y, inf_cordinate_y, rec_cordinate_y, valuesSEIR_x


TuneDataset()
CountrySEIR(21000000)

''' ========================================================== '''

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



