from bertopic import BERTopic
from umap import UMAP
from hdbscan import HDBSCAN
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from typing import List
import plotly.graph_objects as go
from sklearn.decomposition import PCA
import plotly.express as px
import warnings
warnings.filterwarnings("ignore")


def pre_load_bert_model(backend):

    print(f"Preloading BERTopic {backend} backend")

    umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0,
                        metric='cosine', random_state=13)

    hdbscan_model = HDBSCAN(min_cluster_size=60, metric='euclidean',
                        cluster_selection_method='eom', prediction_data=True, 
                        min_samples=10)

    model = BERTopic(verbose=True, embedding_model=backend, 
                    umap_model=umap_model, hdbscan_model=hdbscan_model, 
                    n_gram_range=(1, 3), calculate_probabilities=True)

    mocked_text = ["test "*10 for x in range(10)]

    _, _ = model.fit_transform(mocked_text)


def pre_load_keytotext():
    model = pipeline("mrm8488/t5-base-finetuned-common_gen")
    print(model((["test", "test", "test"])))
    # pipeline("mrm8488/t5-base-finetuned-summarize-news")(["test", "test", "test"])


def visualize_topics(topic_model,
                     topics: List[int] = None,
                     top_n_topics: int = None,
                     width: int = 650,
                     height: int = 650) -> go.Figure:

    freq_df = topic_model.get_topic_freq()
    freq_df = freq_df.loc[freq_df.Topic != -1, :]
    if topics is not None:
        topics = list(topics)
    elif top_n_topics is not None:
        topics = sorted(freq_df.Topic.to_list()[:top_n_topics])
    else:
        topics = sorted(freq_df.Topic.to_list())

    # Extract topic words and their frequencies
    topic_list = sorted(topics)
    frequencies = [topic_model.topic_sizes_[topic] for topic in topic_list]
    words = [" | ".join([word[0] for word in topic_model.get_topic(topic)[:5]]) for topic in topic_list]

    # Embed c-TF-IDF into 2D
    all_topics = sorted(list(topic_model.get_topics().keys()))
    indices = np.array([all_topics.index(topic) for topic in topics])
    embeddings = topic_model.c_tf_idf_.toarray()[indices]
    embeddings = MinMaxScaler().fit_transform(embeddings)

    try:
        embeddings = UMAP(n_neighbors=2, n_components=2, metric='hellinger').fit_transform(embeddings)
    except Exception:
        embeddings = PCA(n_components=2).fit_transform(embeddings)

    # print(type(embeddings))
    # print(embeddings.shape)

    # Visualize with plotly
    df = pd.DataFrame({"x": embeddings[:, 0], "y": embeddings[:, 1],
                       "Topic": topic_list, "Words": words, "Size": frequencies})
    return _plotly_topic_visualization(df, topic_list, width, height)


def _plotly_topic_visualization(df: pd.DataFrame,
                                topic_list: List[str],
                                width: int,
                                height: int):
    """ Create plotly-based visualization of topics with a slider for topic selection """

    def get_color(topic_selected):
        if topic_selected == -1:
            marker_color = ["#B0BEC5" for _ in topic_list]
        else:
            marker_color = ["red" if topic == topic_selected else "#B0BEC5" for topic in topic_list]
        return [{'marker.color': [marker_color]}]

    # Prepare figure range
    x_range = (df.x.min() - abs((df.x.min()) * .15), df.x.max() + abs((df.x.max()) * .15))
    y_range = (df.y.min() - abs((df.y.min()) * .15), df.y.max() + abs((df.y.max()) * .15))

    # Plot topics
    fig = px.scatter(df, x="x", y="y", size="Size", size_max=40, template="simple_white", labels={"x": "", "y": ""},
                     hover_data={"Topic": True, "Words": True, "Size": True, "x": False, "y": False})
    fig.update_traces(marker=dict(color="#B0BEC5", line=dict(width=2, color='DarkSlateGrey')))

    # Update hover order
    fig.update_traces(hovertemplate="<br>".join(["<b>Topic %{customdata[0]}</b>",
                                                 "Words: %{customdata[1]}",
                                                 "Size: %{customdata[2]}"]))

    # Create a slider for topic selection
    steps = [dict(label=f"Topic {topic}", method="update", args=get_color(topic)) for topic in topic_list]
    sliders = [dict(active=0, pad={"t": 50}, steps=steps)]

    # Stylize layout
    fig.update_layout(
        title={
            'text': "<b>Intertopic Distance Map",
            'y': .95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(
                size=22,
                color="Black")
        },
        width=width,
        height=height,
        hoverlabel=dict(
            bgcolor="white",
            font_size=16,
            font_family="Rockwell"
        ),
        xaxis={"visible": False},
        yaxis={"visible": False},
        sliders=sliders
    )

    # Update axes ranges
    fig.update_xaxes(range=x_range)
    fig.update_yaxes(range=y_range)

    # Add grid in a 'plus' shape
    fig.add_shape(type="line",
                  x0=sum(x_range) / 2, y0=y_range[0], x1=sum(x_range) / 2, y1=y_range[1],
                  line=dict(color="#CFD8DC", width=2))
    fig.add_shape(type="line",
                  x0=x_range[0], y0=sum(y_range) / 2, x1=x_range[1], y1=sum(y_range) / 2,
                  line=dict(color="#9E9E9E", width=2))
    fig.add_annotation(x=x_range[0], y=sum(y_range) / 2, text="D1", showarrow=False, yshift=10)
    fig.add_annotation(y=y_range[1], x=sum(x_range) / 2, text="D2", showarrow=False, xshift=10)
    fig.data = fig.data[::-1]

    return fig

def fix_plots(response):
    ldaPlot = response["topicsVisualization"]["ldaPlot"]
    response["topicsVisualization"] = {}
    response["topicsVisualization"]["hierarchicalClusteringPlot"] = None
    response["topicsVisualization"]["topicsWordsScorePlot"] = None
    response["topicsVisualization"]["topicsSimilarityPlot"] = None
    response["topicsVisualization"]["topicPclustersPlot"] = ldaPlot
    return response