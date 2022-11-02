import abc
import pandas as pd
from bertopic import BERTopic
from umap import UMAP
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer
from gensim.models import LdaMulticore
import numpy as np


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

    def train():
        pass
