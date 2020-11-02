from urllib.request import urlopen as uReq
from bs4 import BeautifulSoup as soup


def scrape_current_updates():
    covid19_gov_lk_LINK = 'https://covid19.gov.lk/'
    webpage = uReq(covid19_gov_lk_LINK)
    page_html = webpage.read()
    webpage.close()

    page_soup = soup(page_html, "html.parser")
    page_containers = page_soup.findAll("div", {"class": "situation-blocks"})

    update_dict = {}

    for element in page_containers:
        confirmedCases = element.findAll("span", {"id": "confirmedCases"})
        activeCases = element.findAll("span", {"id": "activeCases"})
        newCases = element.findAll("span", {"id": "newCases"})
        recoveredCases = element.findAll("span", {"id": "recoveredCases"})
        deathCases = element.findAll("span", {"id": "deathCases"})

        try:
            confirmed = confirmedCases[0].text
            active = activeCases[0].text
            newCase = newCases[0].text
            Recovery = recoveredCases[0].text
            death = deathCases[0].text

        except:
            confirmed = 0
            active = 0
            newCase = 0
            Recovery = 0
            death = 0

        closed = int(Recovery)+int(death)

        update_dict = {'confirmed Cases': confirmed, 'active Cases': active, 'new Cases': newCase,
                       'recovered Cases': Recovery, 'death Cases': death,'closed cases':closed}

    return update_dict


def calc_rates():
    updated_data = scrape_current_updates()

    case = list(updated_data.keys())
    case_value = list(updated_data.values())
    confirmed_Cases = int(case_value[0])
    active_Cases = int(case_value[1])
    new_Cases = int(case_value[2])
    recovered_Cases = int(case_value[3])
    death_Cases = int(case_value[4])

    # COVID-19 Mortality Rate.
    Mortality_Rate = round((death_Cases / confirmed_Cases)*100,1)

    # COVID-19 Recovered rate
    Recovered_rate = round((recovered_Cases / confirmed_Cases)*100,2)

    #  COVID-19 CLOSED Cases rate
    CLOSED_Cases_Rate = round(((recovered_Cases + death_Cases) / confirmed_Cases)*100,2)

    #  COVID-19 Active Cases rate
    Active_Cases_rate = round((active_Cases / confirmed_Cases)*100,2)

    Rate_list = [Recovered_rate,Mortality_Rate,CLOSED_Cases_Rate,Active_Cases_rate]
    return Rate_list
