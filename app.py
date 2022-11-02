from flask import Flask
from flask import request
from flask_restx import Api, Resource, reqparse
from topic_modeling import BERTopicModel
import base64
import pandas as pd
import json
import wget
import os

app = Flask(__name__)
api = Api(app)
parser = reqparse.RequestParser()


bertopic = None
lda = None


class SlowAPI(Resource):

    def post(self):

        json_req = {}
        json_res = {}
        
        content_type = request.headers.get('Content-Type')
        if (content_type == 'application/json'):
            json_req = request.get_json()
        else:
            return 'Content-Type not supported!'


        ids = []
        texts = []
        for doc in json_req["documents"]:
            ids.append(doc["id"])
            texts.append(doc["abstract"])


        print("Started topic modeling with BERTopic")
        topics, probs = bertopic.train(texts)

        json_res["documents"] = []

        for doc_info in zip(ids, probs):
            id, prob_unorm = doc_info
            prob = prob_unorm/prob_unorm.sum()

            document = {
                "id": id,
                "topics": []
            }

            for topic in range(len(prob)):
                topic_info = {
                    "topic_name": topic,
                    "affinity": prob[topic]
                }

                document["topics"].append(topic_info)

            json_res["documents"].append(document)
            

            

        print("Generating the plots")
        plots = bertopic.get_plots(texts)

        for plot_name, plot in plots.items():
            file_name = f"{plot_name}.html"
            plot.write_html(file_name)
            with open(file_name, "rb") as html_plot:
                html_text = html_plot.read()
                encoded_text = base64.b64encode(html_text).decode("utf-8")
                json_res[plot_name] = encoded_text
            os.system(f"rm {file_name}")

        return json_res

class FastAPI(Resource):

    def post(self):

        json_req = {}
        json_res = {}
        
        content_type = request.headers.get('Content-Type')
        if (content_type == 'application/json'):
            json_req = request.get_json()
        else:
            return 'Content-Type not supported!'

        print("Started topic modeling with LDA")

        return {}

if __name__ == "__main__":
    bertopic = BERTopicModel()
    bertopic.load_model()
    api.add_resource(SlowAPI, "/slow")
    api.add_resource(FastAPI, "/fast")
    app.run(host='0.0.0.0')
