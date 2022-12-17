import requests
import wget
import pandas as pd
import os
import base64
import json
import time


mock_data_url = "https://raw.githubusercontent.com/daniele-atzeni/A-Systematic-Review-of-Wi-Fi-and-Machine-Learning-Integration-with-Topic-Modeling-Techniques/main/ML_WIFI_preprocessed.csv"
file_name = wget.download(mock_data_url)

data = pd.read_csv("ML_WIFI_preprocessed.csv")
# data = data.head(1000).reset_index()

os.system("rm ML_WIFI_preprocessed.csv")

json_req = {"documents": [], "keywords": ["machine learning", "wifi"]}

ids = data.index.tolist()
ids = list(map(lambda x: str(x), ids))


for i in range(len(ids)):

    document = {
        "id": ids[i],
        "abstract": data.loc[i, "text"]
    }

    json_req["documents"].append(document)

url = "http://127.0.0.1:5000/fast"


start = time.time()
response = requests.post(url=url, json=json_req)
json_res = response.json() # returns a dict

print()
print(time.time()-start)

with open('response.json', 'w') as fp:
    json.dump(json_res, fp)

# plot_names = [
#     "lda_plot" 
# ]

encoded = json_res["topicsVisualization"]["ldaPlot"]
html_code = base64.b64decode(encoded).decode("utf-8")
with open(f"ldaPlot.html", "w") as html_page:
    html_page.write(html_code)