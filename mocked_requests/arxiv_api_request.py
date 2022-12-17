import json
import xmltodict
from functools import reduce
import requests
import pandas as pd
import os
import time
import base64


api = "fast"
# api = "slow"

def make_request(query, max_results):

    query_text = ""
    for keyword in query:
        query_text += f"all:{keyword}+AND+"
    query_text = query_text[:-5]

    print(query_text)
    request_text = f"http://export.arxiv.org/api/query?search_query={query_text}&start=0&max_results={max_results}"
    print(request_text)

    response = requests.get(request_text)
    
    with open("response.xml", "w") as xml_file:
        xml_file.write(response.text)

    with open("response.xml", "r") as xml_file:
        data_dict = xmltodict.parse(xml_file.read())

    json_data = json.dumps(data_dict)

    with open("response.json", "w") as json_file:
        json_file.write(json_data)


def parse_document(document):
    # get title
    title = None if "title" not in document else document["title"]

    # get abstract
    abstract = None if "summary" not in document else document["summary"].replace("\n", " ").replace("\\", " ")

    # get publication date
    date = None if "published" not in document else document["published"]

    # get the authors
    authors = []

    if "author" in document:
        if type(document["author"]) == list:
            for author in document["author"]:
                if "name" in author:
                    authors.append(author["name"])
        else:
            if "name" in document["author"]:
                authors.append(document["author"]["name"])
    
    if authors == []:
        authors = None
    else:
        authors = reduce (lambda a, x: a+x+", ", authors, "")[:-2]


    # get pdf link

    link = None

    if "link" in document:
        # if link is a list
        if type(document["link"]) == list:
            for link in document["link"]:
                if "@title" in link and link["@title"] == "pdf":
                    link = link["@href"]
        # if is not a list
        elif "@title" in link and link["@title"] == "pdf":
            link = link["@href"]        

    # get doi
    doi = None
    if "arxiv:doi" in document:
        doi = document["arxiv:doi"]["#text"]
    elif "id" in document:
        doi = document["id"]
    else:
        doi = None

    result = {
        "title": title, 
        "abstract": abstract,
        "doi": doi,
        "authors": authors,
        "link": link,
        "date": date
    }

    return result


def parse_result():

    with open('response.json') as json_file:
        data = json.load(json_file)

    results = data["feed"]["entry"]

    documents = []

    for document in results:
        parsed = parse_document(document)
        documents.append(parsed)

    return documents


def convert_to_csv(documents):

    attributes = ["title", "abstract","doi", "authors", "link", "date"]

    data_dict = {
        "title": [],
        "abstract": [],
        "doi": [],
        "authors": [],
        "link": [],
        "date": []
    }
    
    for document in documents:
        for attribute in attributes:
            data_dict[attribute].append(document[attribute])

    return pd.DataFrame(data_dict)
    

query = ["cnn", "machine learning"]

start = time.time()
make_request(query, 2000)
documents = parse_result()
end = time.time()
print(end - start)

print(len(documents))

dataframe = convert_to_csv(documents)

os.system("rm response.xml")
os.system("rm response.json")


dataframe.info()

json_req = {"documents": [], "keywords": query}

ids = dataframe.index.tolist()
ids = list(map(lambda x: str(x), ids))


for i in range(len(ids)):

    document = {
        "id": ids[i],
        "abstract": dataframe.loc[i, "abstract"]
    }

    json_req["documents"].append(document)

with open('request.json', 'w') as fp:
    json.dump(json_req, fp)

url = f"http://127.0.0.1:5000/{api}"

start = time.time()
response = requests.post(url=url, json=json_req)
json_res = response.json() # returns a dict

print()
print(time.time()-start)

with open('response.json', 'w') as fp:
    json.dump(json_res, fp)

if api == "slow":
    for plot_name in json_res["topicsVisualization"].keys():
        encoded = json_res["topicsVisualization"][plot_name]

        if encoded == None:
            continue

        html_code = base64.b64decode(encoded).decode("utf-8")
        with open(f"{plot_name}.html", "w") as html_page:
            html_page.write(html_code)

if api == "fast":
    encoded = json_res["topicsVisualization"]["ldaPlot"]
    html_code = base64.b64decode(encoded).decode("utf-8")
    with open(f"ldaPlot.html", "w") as html_page:
        html_page.write(html_code)