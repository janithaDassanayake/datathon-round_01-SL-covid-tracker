from flask import Flask, render_template, request
import getupdate as update
import SEIR_model_island as SEIR_model
import get_covid_updated_data as sl_data
import train_models as train_model
import math
import pandas as pd
import datetime
import numpy as np
import cluster as clusters
import KNN_Model as KNN

app = Flask(__name__)

new_update_list = update.scrape_current_updates()
case = list(new_update_list.keys())
case_value = list(new_update_list.values())

currentDT_out = datetime.datetime.now()
todaydate = currentDT_out.strftime("%Y-%m-%d")
current_time = currentDT_out.strftime("%H:%M")
current_date_time = [todaydate, current_time]
new_rate_list = update.calc_rates()


def perform_covid_training(filename, covid_island_df, predict_model, Days_forecast):
    dates, Actual_Confirmed_all, regression_models_outputs, prediction_date, last_Confirmed, Actual_Recovered_all, last_Recovery_count = train_model.train_predict_plot(
        filename, covid_island_df, predict_model, Days_forecast)

    original_dates = dates

    if len(dates) > 20:
        dates = dates[-20:]
        Actual_Confirmed = Actual_Confirmed_all[-20:]
        Actual_Recovered = Actual_Recovered_all[-20:]

    Actual_Confirmed_all_list = []
    all_data_recoverd = []
    Actual_Confirmed_all_list.append((Actual_Confirmed, 'false', 'Actual Confirmed', '#000000'))
    all_data_recoverd.append((Actual_Recovered, 'false', 'Actual Recovered', '#00008B'))

    for model_name in regression_models_outputs:
        if len(original_dates) > 20:
            Actual_Confirmed_all_list.append(
                (((regression_models_outputs[model_name])[0])[-20:], "true", 'Predicted Confirmed', '#CC2A1E'))
            all_data_recoverd.append(
                (((regression_models_outputs[model_name])[5])[-20:], "true", 'Predicted Recovered', '#008000'))

        else:
            Actual_Confirmed_all_list.append(
                (((regression_models_outputs[model_name])[0]), "true", 'Predicted Confirmed', '#CC2A1E'))
            all_data_recoverd.append(
                (((regression_models_outputs[model_name])[5]), "true", 'Predicted Recovered', '#008000'))


    Confirmed_prediction_data = []
    Covid_model_evaluation = []
    Recovered_prediction_data = []
    list_of_Confirmed_cases = []
    list_of_Confirmed_cases1 = []
    forcasted_values = []
    forcast_date_List_new = []

    Confirmed_prediction_data.append(("Original Confirmed Cases", int(last_Confirmed)))
    Recovered_prediction_data.append(("Original Recovered Cases", int(last_Recovery_count)))

    for model_name in regression_models_outputs:

        Confirmed_prediction_data.append((model_name, int((regression_models_outputs[model_name])[1])))
        Covid_model_evaluation.append((model_name, (regression_models_outputs[model_name])[2]))
        Recovered_prediction_data.append((model_name, int((regression_models_outputs[model_name])[3])))

        list_of_Confirmed_cases.append(regression_models_outputs[model_name][3])
        forcasted_cases_list = regression_models_outputs[model_name][4]

        for x in range(0, int(Days_forecast)):
            roundforcast = forcasted_cases_list[x]
            roundforcast = round(math.ceil(roundforcast))
            forcasted_values.append(roundforcast)

    list_of_Confirmed_cases1.append(last_Confirmed)
    list_of_Confirmed_cases1.extend(forcasted_values)
    max_Recovered = max(list_of_Confirmed_cases1)
    min_Recovered = min(list_of_Confirmed_cases1)
    max_Recovered = math.floor(max_Recovered)
    min_Recovered = math.ceil(min_Recovered)

    startdate = prediction_date
    for x in range(1, int(Days_forecast) + 1):
        enddate = pd.to_datetime(startdate) + pd.DateOffset(days=x)
        enddate = str(enddate)
        input = enddate.replace(' 00:00:00', '')
        forcast_date_List_new.append(input)

    return Confirmed_prediction_data, Confirmed_prediction_data, prediction_date, dates, Actual_Confirmed_all_list, Actual_Confirmed_all_list, all_data_recoverd, all_data_recoverd, Covid_model_evaluation, Recovered_prediction_data, max_Recovered, min_Recovered, forcast_date_List_new, forcasted_values


sl_all_covid_data = sl_data.get_all_sl_covid_data()
SEIR_model.AdjustDates()
cordinates = SEIR_model.FetchCordinates()
all_district_details_dict = SEIR_model.getAllDistrictInfo()
clusetr_dict = clusters.GetClusterPrediction()

'''============================================================================================================'''


@app.route('/Linear_Regression')
def landing_function():
    file_name = 'COVIDSL.COM Data.csv'
    Prediction_Model_algoritms = ['linear_regression']
    forcastingDays = 5

    df = sl_all_covid_data
    Confirmed_prediction_data, Confirmed_prediction_data, prediction_date, dates, Actual_Confirmed_all_list, Actual_Confirmed_all_list, all_data_recoverd, all_data_recoverd, Covid_model_evaluation, Recovered_prediction_data, max_count, min_count, forcast_date_List_new, forcasted_values = perform_covid_training(
        str(file_name), df, Prediction_Model_algoritms, forcastingDays)

    return render_template('index.html', all_test_evaluations=Covid_model_evaluation, show_results_output="true",
                           file_len=len([]), sl_covid_file=[],
                           len_2=len(Confirmed_prediction_data),
                           all_Confirmed_prediction_data=Confirmed_prediction_data,
                           prediction_result_date=prediction_date, dates=dates,
                           all_Confirmed_data=Actual_Confirmed_all_list,
                           case_values=case_value, current_date_time_list=current_date_time,
                           Recovered_prediction_data_list=Recovered_prediction_data, min_count=min_count,
                           max_count=max_count,
                           new_rate_lists=new_rate_list,
                           forcasting=forcasted_values, forcastingdate=forcast_date_List_new,
                           len3=len(forcast_date_List_new),
                           all_data_recoverd_data=all_data_recoverd,
                           len=len(Actual_Confirmed_all_list))


@app.route('/')
def landing_function2():
    file_name = 'COVIDSL.COM Data.csv'
    Prediction_Model_algoritms = ['linear_regression']
    forcastingDays = 5
    df = sl_all_covid_data

    dates, Actual_Confirmed_all, regression_models_outputs, prediction_date, last_Confirmed, Actual_Recovered_all, last_Recovery_count = train_model.train_predict_plot(
        file_name, df, Prediction_Model_algoritms, forcastingDays)

    predicted_x = cordinates[0]
    predicted_y = cordinates[1]
    current_X = cordinates[2]
    current_y = cordinates[3]

    forcasted_values1 = predicted_y[-1]
    forcasted_values2 = predicted_y[-2]
    forcast_list = [int(round(forcasted_values2)), int(round(forcasted_values1))]

    Actual_Confirmed_all_list = []
    forcast_date_List_new = []
    if len(predicted_x) > 30:
        dates = predicted_x[-30:]
        Actual_Confirmed = current_y[-30:-2]

        precicted_Y = predicted_y[-30:]
        arr_1d_f = np.array(precicted_Y)
        arr_1d_f = arr_1d_f.astype('float32')

        Actual_Confirmed_all_list.append((Actual_Confirmed, 'false', 'Actual Confirmed', '#000000'))
        Actual_Confirmed_all_list.append((arr_1d_f, "true", 'Predicted Confirmed', '#CC2A1E'))
    else:
        Actual_Confirmed_all_list.append((predicted_y, "true", 'Predicted Confirmed', '#CC2A1E'))

    startdate = prediction_date
    for x in range(1, 2 + 1):
        enddate = pd.to_datetime(startdate) + pd.DateOffset(days=x)
        enddate = str(enddate)
        input = enddate.replace(' 00:00:00', '')
        forcast_date_List_new.append(input)


    return render_template('SEIR_country.html', all_test_evaluations=[], show_results_output="true",
                           file_len=len([]), sl_covid_file=[],
                           len_2=len([]),
                           all_Confirmed_prediction_data="",
                           prediction_result_date=prediction_date, dates1=dates,
                           all_Confirmed_data22=Actual_Confirmed_all_list,
                           case_values=case_value, current_date_time_list=current_date_time,
                           Recovered_prediction_data_list=[], min_count="",
                           max_count="",
                           new_rate_lists=new_rate_list,
                           forcasting=forcast_list, forcastingdate=forcast_date_List_new,
                           len3=len([]),
                           all_data_recoverd_data=[],
                           lenvv=len(Actual_Confirmed_all_list))


@app.route('/Districs')
def Distric_process():
    district_files = list(all_district_details_dict.keys())


    return render_template('index2.html', all_test_evaluations=[],
                           show_results_output="false",
                           file_len=len(district_files),
                           sl_covid_file=district_files,
                           len_2=len([]),
                           all_Confirmed_prediction_data=[],
                           prediction_result_date="",
                           dates=[],
                           all_Confirmed_data=[],
                           district_name='',
                           case_values=case_value,
                           current_date_time_list=current_date_time,
                           Recovered_prediction_data_list=[],
                           min_count="",
                           max_count="",
                           new_rate_lists=new_rate_list,
                           forcasting=[],
                           forcastingdate=[],
                           len3=len([]),
                           all_data_recoverd_data=[],
                           len=len([]))





@app.route('/geographical')
def geographical_process():
    district_files = list(all_district_details_dict.keys())


    return render_template('geographical.html', all_test_evaluations=[],
                           show_results_output="false",
                           file_len=len(district_files),
                           sl_covid_file=district_files,
                           len_2=len([]),
                           all_Confirmed_prediction_data=[],
                           prediction_result_date="",
                           dates=[],
                           all_Confirmed_data=[],
                           district_name='',
                           case_values=case_value,
                           current_date_time_list=current_date_time,
                           Recovered_prediction_data_list=[],
                           min_count="",
                           max_count="",
                           new_rate_lists=new_rate_list,
                           forcasting=[],
                           forcastingdate=[],
                           len3=len([]),
                           all_data_recoverd_data=[],
                           len=len([]))





@app.route('/process2', methods=['POST'])
def process2():
    sl_district_file_name = request.form['district_file_name']
    # get predicted two days values
    getNearest_Dict = KNN.getNearstDis(str(sl_district_file_name))
    print(getNearest_Dict.keys())


    district_files = list(all_district_details_dict.keys())

    return render_template('geographical.html',
                           show_results_output="true",
                           file_len=len(district_files),
                           sl_covid_file=district_files,
                           case_values=case_value,
                           district_name = sl_district_file_name,
                           current_date_time_list=current_date_time,
                           new_rate_lists=new_rate_list)



@app.route('/cluster')
def Cluster_process():
    district_files = list(all_district_details_dict.keys())
    predted_date_diula1 = clusetr_dict['Divulapitiya']['date_1']
    Divulapitiya_value_1 = clusetr_dict['Divulapitiya']['prediction_1']
    Diulapitya_increased_count1 = clusetr_dict['Divulapitiya']['increase_day_1']

    predted_date_diula2 = clusetr_dict['Divulapitiya']['date_2']
    Divulapitiya_value_2 = clusetr_dict['Divulapitiya']['prediction_2']
    Diulapitya_increased_count2 = clusetr_dict['Divulapitiya']['increase_day_2']

    '''=============================================='''

    Paliyagoda_date_diula1 = clusetr_dict['Paliyagoda']['date_1']
    Paliyagoda_value_1 = clusetr_dict['Paliyagoda']['prediction_1']

    Paliyagoda_increased_count1 = clusetr_dict['Paliyagoda']['increase_day_1']

    Paliyagoda_date_diula2 = clusetr_dict['Paliyagoda']['date_2']
    Paliyagoda_value_2 = clusetr_dict['Paliyagoda']['prediction_2']
    Paliyagoda_increased_count2 = clusetr_dict['Paliyagoda']['increase_day_2']

    date_list = [Paliyagoda_date_diula1, Paliyagoda_date_diula2]
    paliyagoda_count_list= [int(round(Paliyagoda_value_1)),int(round(Paliyagoda_value_2))]
    paliyagoda_increasd_list=[int(round(Paliyagoda_increased_count1)),int(round(Paliyagoda_increased_count2))]

    #date_list = [Paliyagoda_date_diula1, Paliyagoda_date_diula2]
    Divulapitiya_count_list= [int(round(Divulapitiya_value_1)),int(round(Divulapitiya_value_2))]
    Divulapitiya_increasd_list=[int(round(Diulapitya_increased_count1)),int(round(Diulapitya_increased_count2))]

    return render_template('cluster.html', all_test_evaluations=[],
                           show_results_output="false",
                           file_len=len(district_files),
                           sl_covid_file=district_files,
                           len_2=len(paliyagoda_count_list),
                           all_paliyagoda_prediction_data=paliyagoda_count_list,
                           prediction_result_date="",
                           dates=date_list,
                           all_Confirmed_data=[],
                           case_values=case_value,
                           current_date_time_list=current_date_time,
                           paliyagoda_prediction_incresed=paliyagoda_increasd_list,
                           newCluster_value=Divulapitiya_count_list,
                           dulapitiya_incresed=Divulapitiya_increasd_list,
                           new_rate_lists=new_rate_list,
                           forcasting=[],
                           forcastingdate=[],
                           len3=len([]),
                           all_data_recoverd_data=[],
                           len=len([]))


@app.route('/process', methods=['POST'])
def process():
    sl_district_file_name = request.form['district_file_name']
    # get predicted two days values
    forcasted_values = SEIR_model.all_district_details[str(sl_district_file_name)]['predicted_two_days']

    forcasted_values_list = []
    for count in forcasted_values:
        count = int(round(count))
        forcasted_values_list.append(count)

    # get the predicted two dates

    forcast_date_List = SEIR_model.all_district_details[str(sl_district_file_name)]['predict_x'][-1:-3:-1][::-1]
    district_files = list(all_district_details_dict.keys())

    return render_template('index2.html',
                           show_results_output="true",
                           file_len=len(district_files),
                           sl_covid_file=district_files,
                           case_values=case_value,
                           district_name = sl_district_file_name,
                           current_date_time_list=current_date_time,
                           new_rate_lists=new_rate_list,
                           forcasting=forcasted_values_list,
                           forcastingdate=forcast_date_List,
                           len3=len(forcast_date_List))


# main driver function
if __name__ == '__main__':
    app.run()
