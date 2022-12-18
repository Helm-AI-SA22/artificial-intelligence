from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from functools import reduce

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
            
            topics[topic_id].append(doc["text"])


        for topic_id in topics.keys():
            docs_list = topics[topic_id]
            docs_text = reduce(lambda a, doc: a + doc + " ", docs_list)[:-1]
            topics[topic_id] = docs_text

        self.topics_docs = topics

    
    def get_summary(self):
        pass


    def get_titles(self):

        assert self.topics_docs is not None

        for topic_id in self.topics_docs.keys():
            topic_text = self.topics_docs[topic_id]
            input_ids = self.tokenizer(topic_text, return_tensors="pt").input_ids
            generated_ids = self.model.generate(input_ids)
            title = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            print("\n\n\n")
            print(f"{topic_id} title:")
            print(title)
            print("\n\n\n")