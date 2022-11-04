import abc
import pandas as pd
from bertopic import BERTopic
from umap import UMAP
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer
import numpy as np
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
import os
# nltk.download('wordnet')
nltk.download('omw-1.4')

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class TopicModel:
    
    def __init__(self):
        self.model = None
        self.trained = False

    @abc.abstractmethod
    def load_model(self):
        pass

    @abc.abstractmethod
    def train(self, json_data):
        pass

    @abc.abstractmethod
    def get_plots(self):
        pass


class BERTopicModel(TopicModel):

    def __init__(self):
        super().__init__()

    def load_model(self):
        umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0,
                        metric='cosine', random_state=13)

        hdbscan_model = HDBSCAN(min_cluster_size=60, metric='euclidean',
                            cluster_selection_method='eom', prediction_data=True, 
                            min_samples=10)

        self.model = BERTopic(verbose=True, embedding_model="all-MiniLM-L6-v2", 
                        umap_model=umap_model, hdbscan_model=hdbscan_model, 
                        n_gram_range=(1, 3), calculate_probabilities=True)

    def train(self, texts):
        topics, probs = self.model.fit_transform(texts)
        names = self.model.generate_topic_labels(5, False, None, "-")
        self.trained = True
        return topics, probs, names[1:]

    def get_plots(self, texts):
        plots = {}
        if self.trained:
            plots["topic_clusters_plot"] = self.model.visualize_topics()
            plots["hierarchical_clustering_plot"] = self.model.visualize_hierarchy()
            plots["topics_words_score_plot"] = self.model.visualize_barchart()
            plots["topics_similarity_plot"] = self.model.visualize_heatmap()
            # plots["document_clusters_plot"] = self.model.visualize_documents(texts)
        return plots


class LDAModel(TopicModel):

    def __init__(self, k):
        super().__init__()
        self.k = k


    def get_topics_num(self, texts):
        
        frac = 0.25

        n = int(len(texts)*frac)

        texts_series = pd.Series(texts)
        texts_series = texts_series.sample(n)


        umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0,
                        metric='cosine', random_state=13)

        hdbscan_model = HDBSCAN(min_cluster_size=60, metric='euclidean',
                            cluster_selection_method='eom', prediction_data=True, 
                            min_samples=10)

        infering_model = BERTopic(verbose=True, embedding_model="paraphrase-MiniLM-L3-v2", 
                        umap_model=umap_model, hdbscan_model=hdbscan_model, n_gram_range=(1, 3))

        topics, _ = infering_model.fit_transform(texts)

        return len(set(topics))
    
    def train(self, texts):

        def lemmatize_stemming(text):
            stemmer = SnowballStemmer('english')
            return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))

        def preprocess(text):
            result = []
            for token in gensim.utils.simple_preprocess(text):
                if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
                    result.append(lemmatize_stemming(token))
            return result


        documents = pd.Series(texts)

        # lemmatization, stemming and stopword removal
        processed_docs = documents.map(preprocess)

        dictionary = gensim.corpora.Dictionary(processed_docs)
        # possible filtering 
        # dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=100000)

        bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]


        print("PROVA1")
        self.k = self.get_topics_num(texts)
        print("PROVA2")

        lda_model = gensim.models.LdaMulticore(bow_corpus, num_topics=self.k, id2word=dictionary,
                                                passes=2, workers=2, minimum_probability=0.0)

        self.model = lda_model
        self.id2word = dictionary
        self.corupus = bow_corpus

        # for idx, topic in lda_model.print_topics(-1):
        #     print('Topic: {} \nWords: {}'.format(idx, topic))

        probabilities = []

        for i in range(len(bow_corpus)):
            probabilities.append(lda_model[bow_corpus[i]])

        topn = 5
        names = []
        for i in range(self.k):
            terms = lda_model.get_topic_terms(i, topn=topn)
            topic_title = ""
            for term in terms:
                term_id = term[0]
                word = dictionary[term_id]
                topic_title += word + "-"
            names.append(topic_title[:-1])
                

        return probabilities, names

    def get_plots(self):
        p = pyLDAvis.gensim_models.prepare(self.model, self.corupus, self.id2word)
        pyLDAvis.save_html(p, 'lda_plot.html')
