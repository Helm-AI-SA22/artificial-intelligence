import pandas as pd
from bertopic import BERTopic
from umap import UMAP
from hdbscan import HDBSCAN
import numpy as np
import time 
import wget
import os

umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0,
                        metric='cosine', random_state=13)

hdbscan_model = HDBSCAN(min_cluster_size=60, metric='euclidean',
                    cluster_selection_method='eom', prediction_data=True, 
                    min_samples=10)

model = BERTopic(verbose=True, embedding_model="paraphrase-MiniLM-L3-v2", 
                umap_model=umap_model, hdbscan_model=hdbscan_model, 
                n_gram_range=(1, 3))

mock_data_url = "https://raw.githubusercontent.com/daniele-atzeni/A-Systematic-Review-of-Wi-Fi-and-Machine-Learning-Integration-with-Topic-Modeling-Techniques/main/ML_WIFI_preprocessed.csv"
file_name = wget.download(mock_data_url)

data = pd.read_csv("ML_WIFI_preprocessed.csv")
data = data.sample(350)
data.info()

os.system("rm ML_WIFI_preprocessed.csv")

texts = data["text"]

start = time.time()
topics, _ = model.fit_transform(texts)
print(set(topics))
print(time.time()-start)