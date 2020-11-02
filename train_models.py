import utils



def train_predict_plot(file_name, covid_df, model_list, forcastingDays):
    prediction_models_outputs = {}

    dates, comfermd, test_date, test_covid_comfirem, reciverySet, today_Predicted_recover = utils.get_sl_covid_Data(
        covid_df)

    for model in model_list:
        method_to_call = getattr(utils, model)
        prediction_models_outputs[model] = method_to_call(dates, comfermd, test_date, covid_df, forcastingDays)

    dates = list(covid_df['Date'])
    predict_date = dates[-1]

    dates = dates

    return dates, comfermd, prediction_models_outputs, predict_date, test_covid_comfirem, reciverySet, today_Predicted_recover
