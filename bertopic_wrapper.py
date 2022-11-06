import numpy as np
import pandas as pd
from typing import List, Union, Tuple
from bertopic.backend._utils import select_backend
from bertopic._utils import MyLogger, check_documents_type, check_embeddings_shape, check_is_fitted
from bertopic import BERTopic
from sklearn.decomposition import PCA

class BERTopicWrapper(BERTopic):
    def fit_transform(self,
                      documents: List[str],
                      embeddings: np.ndarray = None,
                      y: Union[List[int], np.ndarray] = None) -> Tuple[List[int],
                                                                       Union[np.ndarray, None]]:

        check_documents_type(documents)
        check_embeddings_shape(embeddings, documents)
        self.temp_umap = None

        documents = pd.DataFrame({"Document": documents,
                                  "ID": range(len(documents)),
                                  "Topic": None})

        # Extract embeddings
        if embeddings is None:
            self.embedding_model = select_backend(self.embedding_model,
                                                  language=self.language)
            embeddings = self._extract_embeddings(documents.Document,
                                                  method="document",
                                                  verbose=self.verbose)
            # logger.info("Transformed documents to Embeddings")
        else:
            if self.embedding_model is not None:
                self.embedding_model = select_backend(self.embedding_model,
                                                      language=self.language)

        # Reduce dimensionality with UMAP
        try:
            if self.seed_topic_list is not None and self.embedding_model is not None:
                y, embeddings = self._guided_topic_modeling(embeddings)
            umap_embeddings = self._reduce_dimensionality(embeddings, y)
            print("Executed dimensionality reduction with UMAP")
        except Exception:
            self.temp_umap = self.umap
            self.umap = PCA(n_components=2)
            if self.seed_topic_list is not None and self.embedding_model is not None:
                y, embeddings = self._guided_topic_modeling(embeddings)
                print("Executed dimensionality reduction with PCA")
            umap_embeddings = self._reduce_dimensionality(embeddings, y)

        # Cluster UMAP embeddings with HDBSCAN
        documents, probabilities = self._cluster_embeddings(umap_embeddings, documents)

        # Sort and Map Topic IDs by their frequency
        if not self.nr_topics:
            documents = self._sort_mappings_by_frequency(documents)

        # Extract topics by calculating c-TF-IDF
        self._extract_topics(documents)

        # Reduce topics
        if self.nr_topics:
            documents = self._reduce_topics(documents)

        self._map_representative_docs(original_topics=True)
        probabilities = self._map_probabilities(probabilities, original_topics=True)
        predictions = documents.Topic.to_list()

        self.sentence_embeddings = embeddings
        self.umap_embeddings = umap_embeddings
        
        self.umap = self.temp_umap

        return predictions, probabilities