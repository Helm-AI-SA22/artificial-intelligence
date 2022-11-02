import requests
import wget
import pandas as pd
import os
import base64
import json
from gensim.models import LdaMulticore
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
import numpy as np
import pyLDAvis
import pyLDAvis.gensim_models
import nltk
# nltk.download('wordnet')
nltk.download('omw-1.4')

def lemmatize_stemming(text):
    stemmer = SnowballStemmer('english')
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))

def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(lemmatize_stemming(token))
    return result

mock_data_url = "https://raw.githubusercontent.com/daniele-atzeni/A-Systematic-Review-of-Wi-Fi-and-Machine-Learning-Integration-with-Topic-Modeling-Techniques/main/ML_WIFI_preprocessed.csv"
file_name = wget.download(mock_data_url)

data = pd.read_csv("ML_WIFI_preprocessed.csv")
# data = data.head(1000)

os.system("rm ML_WIFI_preprocessed.csv")

documents = data["text"]


# lemmatization, stemming and stopword removal
processed_docs = documents.map(preprocess)

dictionary = gensim.corpora.Dictionary(processed_docs)
# possible filtering 
# dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=100000)

bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]
lda_model = gensim.models.LdaMulticore(bow_corpus, num_topics=5, id2word=dictionary, passes=2, workers=2)

for idx, topic in lda_model.print_topics(-1):
    print('Topic: {} \nWords: {}'.format(idx, topic))

p = pyLDAvis.gensim_models.prepare(lda_model, bow_corpus, dictionary)
pyLDAvis.save_html(p, 'lda_plot.html')