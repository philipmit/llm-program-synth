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
