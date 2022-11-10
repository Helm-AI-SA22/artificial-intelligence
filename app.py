from flask import Flask
from flask import request
from flask_restx import Api, Resource, reqparse
from topic_modeling import BERTopicModel, LDAModel
from utils import pre_load_bert_model, fix_plots
from handler import fast_api_handler, slow_api_handler
from flask_cors import CORS
import os
os.environ["SENTENCE_TRANSFORMERS_HOME"] = f"{os.getcwd()}/.cache/"
import warnings
warnings.filterwarnings("ignore")


app = Flask(__name__)
cors = CORS(app, resources={r"/*": {"origins": "*"}})

api = Api(app)
parser = reqparse.RequestParser()


bertopic = None
lda = None


class SlowAPI(Resource):

    def post(self):

        json_req = {}
        
        content_type = request.headers.get('Content-Type')
        if (content_type == 'application/json'):
            json_req = request.get_json()
        else:
            return 'Content-Type not supported!'
            
        try:
            return slow_api_handler(json_req, bertopic)
        except Exception:
            try:
                print("Slow topic modeling failed, switching to fast")
                response =  fast_api_handler(json_req, lda)
                return fix_plots(response)
            except Exception:
                print("Fast topic modeling failed, switching to LDA with fixed topic number")
                response = fast_api_handler(json_req, lda, 2)
                return fix_plots(response)



class FastAPI(Resource):

    def post(self):

        json_req = {}
        
        content_type = request.headers.get('Content-Type')
        if (content_type == 'application/json'):
            json_req = request.get_json()
        else:
            return 'Content-Type not supported!'
        
        try:
            return fast_api_handler(json_req, lda)
        except Exception:
            print("Fast topic modeling failed, switching to LDA with fixed topic number")
            return fast_api_handler(json_req, lda, 2)
        

if __name__ == "__main__":

    pre_load_bert_model("all-MiniLM-L6-v2")
    pre_load_bert_model("paraphrase-MiniLM-L3-v2")

    bertopic = BERTopicModel()
    lda = LDAModel()

    bertopic.load_model()
    lda.load_model()

    api.add_resource(SlowAPI, "/slow")
    api.add_resource(FastAPI, "/fast")
    app.run(host='0.0.0.0')