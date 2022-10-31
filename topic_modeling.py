import abc
import pandas as pd
from bertopic import BERTopic
from umap import UMAP
from hdbscan import HDBSCAN


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

    def train(self, json_data):
        data = pd.DataFrame(json_data)
        topics, probs = self.model.fit_transform(data['text'])
        self.trained = True
        return topics, probs

    def get_plots(self):
        plots = {}
        plots["topics"] = self.model.visualize_topics()
        plots["hierarchy"] = self.model.visualize_hierarchy()
        plots["terms_score"] = self.model.visualize_barchart()
        return plots


class LDAModel(TopicModel):

    def __init__(self):
        super().__init__()
