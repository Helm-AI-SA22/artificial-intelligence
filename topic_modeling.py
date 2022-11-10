import abc
import pandas as pd
from bertopic import BERTopic
from utils import visualize_topics
from umap import UMAP
from hdbscan import HDBSCAN
from gensim.models import LdaMulticore
import gensim
from functools import reduce
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
import pyLDAvis
import pyLDAvis.gensim_models
from sklearn.decomposition import PCA
import nltk
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
nltk.download('omw-1.4')
nltk.download('wordnet')

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

    def preprocess(self, text):

        def lemmatize(text):
            return WordNetLemmatizer().lemmatize(text, pos='v')

        def process(text):
            result = []
            for token in gensim.utils.simple_preprocess(text):
                if token not in gensim.parsing.preprocessing.STOPWORDS: #and len(token) > 3:
                    result.append(lemmatize(token))
            return result


        text = pd.Series(text)
        return text.map(process)



class BERTopicModel(TopicModel):

    def __init__(self):
        super().__init__()

    def load_model(self):
        umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0,
                        metric='cosine', random_state=13)
        # umap_model = PCA(n_components=5)

        hdbscan_model = HDBSCAN(min_cluster_size=60, metric='euclidean',
                            cluster_selection_method='eom', prediction_data=True, 
                            min_samples=10)

        self.model = BERTopic(verbose=True, embedding_model="all-MiniLM-L6-v2", 
                        umap_model=umap_model, hdbscan_model=hdbscan_model, 
                        n_gram_range=(1, 3), calculate_probabilities=True)


    def train(self, texts):

        texts = self.preprocess(texts).map(lambda t: reduce(lambda a, x: a + x + " ", t, "")[:-1]).tolist()

        topics, probs = self.model.fit_transform(texts)
        names = self.model.generate_topic_labels(5, False, None, "-")
        self.trained = True

        all_topics = list(set(topics))

        if -1 in all_topics:
            all_topics.remove(-1)
            names = names[1:]
                    
        self.num_topics = len(all_topics)

        return topics, probs, names


    def get_plots(self):

        assert self.trained == True

        def create_plot(plot_func, *params):
            try:
                return plot_func(*params)
            except Exception:
                return None

        plots = {}

        plots["hierarchical_clustering_plot"] = create_plot(self.model.visualize_hierarchy)
        plots["topics_words_score_plot"] = create_plot(self.model.visualize_barchart, None, self.num_topics)
        plots["topics_similarity_plot"] = create_plot(self.model.visualize_heatmap)
        plots["topic_clusters_plot"] = create_plot(visualize_topics, self.model)

        return plots





class LDAModel(TopicModel):

    def __init__(self):
        super().__init__()


    def load_model(self):
        umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0,
                        metric='cosine', random_state=13)
        # umap_model = PCA(n_components=5)

        hdbscan_model = HDBSCAN(min_cluster_size=60, metric='euclidean',
                            cluster_selection_method='eom', prediction_data=True, 
                            min_samples=10)

        self.inferring_model = BERTopic(verbose=True, embedding_model="paraphrase-MiniLM-L3-v2", 
                                umap_model=umap_model, hdbscan_model=hdbscan_model,
                                n_gram_range=(1, 3))


    def get_topics_num(self, texts):
        
        frac = 0.25

        n = int(len(texts)*frac)

        texts_series = pd.Series(texts)
        texts_series = texts_series.sample(n).tolist()

        topics, _ = self.inferring_model.fit_transform(texts_series)

        # self.topics = topics

        return len(set(topics))

    
    def train(self, texts, num_topics = None):

        # lemmatization, stemming and stopword removal
        processed_docs = self.preprocess(texts)

        dictionary = gensim.corpora.Dictionary(processed_docs)
        # possible filtering 
        # dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=100000)

        bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]


        if num_topics is None:
            self.num_topics = self.get_topics_num(texts)
            self.num_topics = self.num_topics if self.num_topics > 0 else 1
        else:
            self.num_topics = num_topics

        lda_model = LdaMulticore(bow_corpus, num_topics=self.num_topics, id2word=dictionary,
                                passes=2, workers=2, minimum_probability=0.0)

        self.trained = True
        self.model = lda_model
        self.id2word = dictionary
        self.corupus = bow_corpus

        probabilities = []

        for i in range(len(bow_corpus)):
            probabilities.append(lda_model[bow_corpus[i]])

        topn = 5
        names = []
        for i in range(self.num_topics):
            terms = lda_model.get_topic_terms(i, topn=topn)
            topic_title = ""
            for term in terms:
                term_id = term[0]
                word = dictionary[term_id]
                topic_title += word + "-"
            names.append(topic_title[:-1])
                

        return probabilities, names


    def get_plots(self):
        
        assert self.trained == True

        p = pyLDAvis.gensim_models.prepare(self.model, self.corupus, self.id2word)
        pyLDAvis.save_html(p, 'lda_plot.html')