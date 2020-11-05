import requests as rs
import json
import pandas as pd
import datetime
import getupdate as update

def get_all_sl_covid_data():
    url = "https://disease.sh/v3/covid-19/historical/Sri%20Lanka?lastdays=200"
    response = rs.get(url)
    data = response.text
    parsed = dict(json.loads(data))

    # print(json.dumps(parsed, indent=4))
    df = pd.DataFrame(parsed["timeline"]).reset_index()

    df['Date'] = df['index'].apply(lambda x: datetime.datetime.strptime(x, '%m/%d/%y').strftime('%Y-%m-%d'))
    df_new = df[["Date", "cases", "deaths", "recovered"]]

    # res = rs.get('https://disease.sh/v3/covid-19/countries/Sri%20Lanka')
    # text = res.text
    # parsed = json.loads(text)
    # current = dict(parsed)

    updates = update.scrape_current_updates()
    updates_list = list(updates.values())

    row = {"Date": datetime.datetime.now().strftime('%Y-%m-%d'), "cases": int(updates_list[0]), "deaths":int(updates_list[4]),
           "recovered":int(updates_list[3])}

    df_new = df_new.append(row, ignore_index=True)

    df_new = df_new.rename(columns={"cases": "Confirmed", "deaths": "Critical", "recovered": "Recovered"})

    return df_new
