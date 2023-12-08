#<PrevData>
print('********** Load and preview the dataset and datatypes')
##### define variables based on prompt
variable_text='the percentage of people with coronary heart disease'
time1='2016'
time2='2021'
city='Philadelphia'


##### collect data from Data Commons
import requests
import json
query=variable_text+' in '+city
url = f"https://datacommons.org/api/explore/detect?q={query}"
headers = {
    "Content-Type": "application/json"
}
data = {
    "contextHistory": [],
    "dc": ""
}
response = requests.post(url, headers=headers, json=data)
res_data = json.loads(response.text)
entity = res_data['entities'][0]
variable = res_data['variables'][0]
url = "https://api.datacommons.org/v2/observation"
params = {
"key": "AIzaSyDZ7TUirJBt2BSVf7jNuPpk29XzvAeeONI",
"entity.dcids": entity, # e.g. "country/USA",
"select": ["entity", "variable", "value", "date"],
"variable.dcids": variable
}
response = requests.get(url, params=params)
print('response.text')
print(response.text)
#</PrevData>
#<PrepData>
print('********** Prepare the dataset for analysis and visualization')
##### extract the data from the response
obs_data = json.loads(response.text)
print('obs_data')
print(obs_data)
entity_data = obs_data["byVariable"][variable]["byEntity"][entity]
print('entity_data')
print(entity_data)
print('type(entity_data)')
print(type(entity_data))
seq_data = entity_data["orderedFacets"][0]['observations']
print('seq_data')
print(seq_data)
print('type(seq_data)')
print(type(seq_data))
print('len(seq_data)')
print(len(seq_data))
#</PrepData>
#<Analysis>
from scipy import stats
import numpy as np
print('********** Perform the analysis to answer the question in the prompt')
##### extract data from the timeframe specified in the prompt
seq_data_in_timeframe=[x for x in seq_data if int(x['date']) >= int(time1) and int(x['date']) <= int(time2)]
print('seq_data_in_timeframe')
print(seq_data_in_timeframe)
print('len(seq_data_in_timeframe)')
print(len(seq_data_in_timeframe))
X=[int(x['date']) for x in seq_data_in_timeframe]
print('X')
print(X)
y=[x['value'] for x in seq_data_in_timeframe]
print('y')
print(y)
slope, intercept, r_value, p_value, std_err = stats.linregress(X, y)
print('slope')
print(slope)
##### the answer options are 'increasing' or 'decreasing'
if slope>0:
    natural_language_answer='increasing'
else:
    natural_language_answer='decreasing'
print(natural_language_answer)
#</Analysis>
