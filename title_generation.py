from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from functools import reduce
import random

class TitleGenerator:


    def __init__(self):
        self.model = AutoModelForSeq2SeqLM.from_pretrained("Callidior/bert2bert-base-arxiv-titlegen")
        self.tokenizer = AutoTokenizer.from_pretrained("Callidior/bert2bert-base-arxiv-titlegen")
        self.topics_docs = None


    def set_documents(self, response):
        topic_ids = list(filter(lambda tid: tid != -1, map(lambda topic: topic["id"], response["topics"])))
        topics = {}

        for tid in topic_ids:
            topics[tid] = []

        for doc in response["documents"]:
            topics_list = list(map(lambda topic: (topic["id"], topic["affinity"]), doc["topics"]))
            topic_id = reduce(lambda a, x: x if x[1] > a[1] else a, topics_list)[0]

            if topic_id == -1:
                continue
            
            doc_info = {
                "text": doc["text"],
                "affinity": doc["topics"][topic_id]["affinity"]
            }

            topics[topic_id].append(doc_info)


        for topic_id in topics.keys():
            topics[topic_id].sort(reverse=True, key=lambda doc: doc["affinity"])
            topics[topic_id] = list(map(lambda x: x["text"], topics[topic_id]))

        self.topics_docs = topics


    def get_titles(self, k=10):

        assert self.topics_docs is not None

        topics_titles = {}

        for topic_id in self.topics_docs.keys():

            print(f"Generating titles for topic {topic_id}")

            topics_titles[topic_id] = {
                "summary": [],
            }

            topics_titles_text = ""

            topic_k = min(k, len(self.topics_docs[topic_id]))

            topic_texts = self.topics_docs[topic_id]
            for topic_text in random.sample(topic_texts, topic_k):

                input_ids = self.tokenizer(topic_text, return_tensors="pt", max_length=512, truncation=True).input_ids
                generated_ids = self.model.generate(input_ids)
                generated_title = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

                topics_titles[topic_id]["summary"].append(generated_title)
                topics_titles_text += generated_title + ". "

            input_ids = self.tokenizer(topics_titles_text[:-1], return_tensors="pt", max_length=512, truncation=True).input_ids
            generated_ids = self.model.generate(input_ids)
            generated_title = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

            topics_titles[topic_id]["title"] = generated_title

        return topics_titles