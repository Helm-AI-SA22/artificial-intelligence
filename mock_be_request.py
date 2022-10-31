import requests
import wget
import pandas as pd
import os
import base64


mock_data_url = "https://raw.githubusercontent.com/daniele-atzeni/A-Systematic-Review-of-Wi-Fi-and-Machine-Learning-Integration-with-Topic-Modeling-Techniques/main/ML_WIFI_preprocessed.csv"
file_name = wget.download(mock_data_url)

data = pd.read_csv("ML_WIFI_preprocessed.csv")
data = data.head(1000)

print(data.head())

os.system("rm ML_WIFI_preprocessed.csv")

json_req = {"document_ids": [], "text": []}

ids = data.index.tolist()


for i in range(len(ids)):
    json_req["document_ids"].append(ids[i])
    json_req["text"].append(data.loc[i, "text"])


url = "http://127.0.0.1:5000/modeling?model=bertopic"
response = requests.post(url=url, json=json_req)
json_res = response.json() # returns a dict

html_code = base64.b64decode(json_res["plots"]["topics"]).decode("utf-8")
with open("test.html", "w") as html_page:
    html_page.write(html_code)
