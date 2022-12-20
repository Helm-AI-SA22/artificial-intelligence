from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, LEDForConditionalGeneration, LEDConfig
from functools import reduce
import json
import torch
import random
import os
import time
os.environ["SENTENCE_TRANSFORMERS_HOME"] = f"{os.getcwd()}/.cache/"
os.environ["TRANSFORMERS_CACHE"] = f"{os.getcwd()}/.cache/"
os.environ["PYTORCH_TRANSFORMERS_CACHE"] = f"{os.getcwd()}/.cache/"
os.environ["HF_HOME"] = f"{os.getcwd()}/.cache/"
os.environ["PYTORCH_PRETRAINED_BERT_CACHE"] = f"{os.getcwd()}/.cache/"

class TitleGenerator:


    def __init__(self):
        self.model = AutoModelForSeq2SeqLM.from_pretrained("Callidior/bert2bert-base-arxiv-titlegen")
        self.tokenizer = AutoTokenizer.from_pretrained("Callidior/bert2bert-base-arxiv-titlegen")
        # self.tokenizer = AutoTokenizer.from_pretrained("allenai/PRIMERA-arxiv")
        # self.model = AutoModelForSeq2SeqLM.from_pretrained("allenai/PRIMERA-arxiv")

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
            print(len(docs_list))

        self.topics_docs = topics


    def get_titles(self):
        assert self.topics_docs is not None

        for topic_id in self.topics_docs.keys():

            start = time.time()

            topics_titles = ""

            topic_texts = self.topics_docs[topic_id]
            for topic_text in random.sample(topic_texts, 3):
                input_ids = self.tokenizer(topic_text, return_tensors="pt", max_length=512, truncation=True).input_ids
                generated_ids = self.model.generate(input_ids)
                generated_title = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
                topics_titles += generated_title + ". "

            print("\n\n\n")
            print(f"{topic_id} title:")
            print(topics_titles)
            input_ids = self.tokenizer(topics_titles[:-1], return_tensors="pt", max_length=512, truncation=True).input_ids
            generated_ids = self.model.generate(input_ids)
            generated_title = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            print("\n")
            print(generated_title)
            end = time.time()
            print(end-start)
            print("\n\n\n")

            


    def get_summary(self):

        DOCSEP_TOKEN_ID = self.tokenizer.convert_tokens_to_ids("<doc-sep>")
        PAD_TOKEN_ID = self.tokenizer.pad_token_id

        input_ids_all = []
        

        for topic_id in self.topics_docs.keys():
            

            print(f"Processing topic {topic_id}")
            
            input_ids = []

            for doc in self.topics_docs[topic_id][:2]:
                input_ids.extend(
                    self.tokenizer.encode(
                        doc,
                        truncation=True,
                        max_length=4096 // len(self.topics_docs[topic_id]),
                    )[1:-1]
                )
                input_ids.append(DOCSEP_TOKEN_ID)
            input_ids_all.append(torch.tensor(input_ids))
            
            
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids_all, batch_first=True, padding_value=PAD_TOKEN_ID
        )

        # get the input ids and attention masks together
        global_attention_mask = torch.zeros_like(input_ids).to(input_ids.device)

        global_attention_mask[:, 0] = 1
        global_attention_mask[input_ids == DOCSEP_TOKEN_ID] = 1
        generated_ids = self.model.generate(
            input_ids=input_ids,
            global_attention_mask=global_attention_mask,
            use_cache=True,
            max_length=1024,
            num_beams=5,
        )
        generated_str = self.tokenizer.batch_decode(
            generated_ids.tolist(), skip_special_tokens=True
        )

        print(generated_str)

                



with open('example.json') as json_file:
    data = json.load(json_file)

tg = TitleGenerator()
tg.set_documents(data)
del data
tg.get_titles()
# summaries = tg.get_summary()