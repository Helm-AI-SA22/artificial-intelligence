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
data = data.head(1000).reset_index()

# data.info()

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

with open('response.json', 'w') as fp:
    json.dump(json_req, fp)

url = "http://127.0.0.1:5000/slow"

start = time.time()
response = requests.post(url=url, json=json_req)
json_res = response.json() # returns a dict

# print(json_res)

print()
print(time.time()-start)

with open('response.json', 'w') as fp:
    json.dump(json_res, fp)

for plot_name in json_res["topicsVisualization"].keys():

    encoded = json_res["topicsVisualization"][plot_name]
    print(encoded[:10])

    if encoded == None:
        continue

    html_code = base64.b64decode(encoded).decode("utf-8")
    with open(f"{plot_name}.html", "w") as html_page:
        html_page.write(html_code)


# for plot_name in json_res["topicsVisualization"].keys():

#     encoded = json_res["topicsVisualization"][plot_name]
#     print(encoded[:10])

#     if encoded == None:
#         continue

#     if plot_name.endswith("Interactive"):
#         html_code = base64.b64decode(encoded).decode("utf-8")
#         with open(f"{plot_name}.html", "w") as html_page:
#             html_page.write(html_code)

#     if plot_name.endswith("Static"):
#         image_code = base64.b64decode(encoded)
#         with open(f"{plot_name}.png", "bw") as png_image:
#             png_image.write(image_code)