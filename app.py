from flask import Flask
from flask import request
from flask_restx import Api, Resource, reqparse
from topic_modeling import BERTopicModel, LDAModel
from utils import pre_load_bert_model
from handler import fast_api_handler, slow_api_handler
from flask_cors import CORS
import os
os.environ["SENTENCE_TRANSFORMERS_HOME"] = f"{os.getcwd()}/.cache/"


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
            print("Error during topic modeling. Switching to fast API")
            try:
                return fast_api_handler(json_req, lda)
            except Exception:
                print("Error during topic modeling. ")
                return fast_api_handler(json_req, lda, 2)



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