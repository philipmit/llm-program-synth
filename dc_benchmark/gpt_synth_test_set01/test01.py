#<PrevData>
print('********** Load and preview the dataset and datatypes')
##### define variables based on prompt
variable_text='the prevalence of chronic kidney disease'
time='2014'
city1='Seattle'
city2='Boise'


##### collect data from Data Commons
import requests
import json
query1=variable_text+' in '+city1
query2=variable_text+' in '+city2
url = f"https://datacommons.org/api/explore/detect?q={query1}"
headers = {
    "Content-Type": "application/json"
}
data = {
    "contextHistory": [],
    "dc": ""
}
response1 = requests.post(url, headers=headers, json=data)
res_data1 = json.loads(response1.text)
entity1 = res_data1['entities'][0]
variable = res_data1['variables'][0]

url = f"https://datacommons.org/api/explore/detect?q={query2}"
response2 = requests.post(url, headers=headers, json=data)
res_data2 = json.loads(response2.text)
entity2 = res_data2['entities'][0]

url = "https://api.datacommons.org/v2/observation"
params = {
"key": "AIzaSyDZ7TUirJBt2BSVf7jNuPpk29XzvAeeONI",
"entity.dcids": [entity1, entity2], # e.g. "country/USA",
"select": ["entity", "variable", "value", "date"],
"variable.dcids": variable
}
response = requests.get(url, params=params)
print(response.text)
#</PrevData>
