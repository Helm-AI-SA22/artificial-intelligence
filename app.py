from flask import Flask
from flask import request
from flask_restful import Api, Resource, reqparse
from topic_modeling import BERTopicModel
import base64
import pandas as pd
import json
import wget

app = Flask(__name__)
api = Api(app)
parser = reqparse.RequestParser()


bertopic = None
lda = None


class AIServer(Resource):

    def get(self):
        if request.method == "GET":
            return "<h1>TEST</h1>"

    def post(self):

        print("Request arrive")

        json_req = {}
        json_res = {}
        
        content_type = request.headers.get('Content-Type')
        if (content_type == 'application/json'):
            json_req = request.get_json()
        else:
            return 'Content-Type not supported!'

        model_name = request.values.get("model")

        if model_name not in ["bertopic", "lda"]:
            return 'Model not supported!'

        if model_name == "bertopic":
            topic_model = bertopic

        print("Started topic modeling")
        topics, probs = topic_model.train(json_req)

        print(type(topics))

        json_res["document_ids"] = json_req["document_ids"]
        json_res["topics"] = topics

        plots = topic_model.get_plots()


        json_res["plots"] = {}
        for plot_name, plot in plots.items():
            file_name = f"{plot_name}.html"
            plot.write_html(file_name)
            with open(file_name, "r") as html_plot:
                html_text = html_plot.read().encode("ascii")
                encoded_text = base64.b64encode(html_text)
                print(type(encoded_text))
                json_res["plots"][plot_name] = str(encoded_text)

        with open("res.json", "w") as res:
            res.write(json.dumps(json_res, indent=4))

        return json_res



if __name__ == "__main__":
    bertopic = BERTopicModel()
    bertopic.load_model()
    api.add_resource(AIServer, "/modeling")
    app.run(host='0.0.0.0')
