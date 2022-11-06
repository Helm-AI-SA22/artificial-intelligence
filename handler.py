import os
import base64

def slow_api_handler(json_req, model):

    json_res = {}

    ids = []
    texts = []
    for doc in json_req["documents"]:
        ids.append(doc["id"])
        texts.append(doc["abstract"])


    print("Started topic modeling with BERTopic")
    topics, probs, names = model.train(texts)

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
                "id": topic,
                "affinity": prob[topic]
            }

            document["topics"].append(topic_info)

        json_res["documents"].append(document)


    json_res["topics"] = []

    for i in range(len(names)):
        topic_name = names[i]
        topic = {
            "id": i,
            "name": topic_name
        }
        json_res["topics"].append(topic)

    print("Generating the plots")
    plots = model.get_plots()

    for plot_name, plot in plots.items():
        
        if plot == None:
            json_res[plot_name] = None
            continue

        file_name = f"{plot_name}.html"
        plot.write_html(file_name)
        with open(file_name, "rb") as html_plot:
            html_text = html_plot.read()
            encoded_text = base64.b64encode(html_text).decode("utf-8")
            json_res[plot_name] = encoded_text
        os.system(f"rm {file_name}")

    return json_res




def fast_api_handler(json_req, model):

    json_res = {}

    ids = []
    texts = []
    for doc in json_req["documents"]:
        ids.append(doc["id"])
        texts.append(doc["abstract"])


    print("Started topic modeling with LDA")
    probs, names = model.train(texts)

    json_res["documents"] = []

    for doc_info in zip(ids, probs):
        id, prob = doc_info

        document = {
            "id": id,
            "topics": []
        }

        for topic in prob:

            topic_info = {
                "id": topic[0], 
                "affinity": float(topic[1])
            }

            document["topics"].append(topic_info)

        json_res["documents"].append(document)
    

    json_res["topics"] = []

    for i in range(len(names)):
        topic_name = names[i]
        topic = {
            "id": i,
            "name": topic_name
        }
        json_res["topics"].append(topic)

    model.get_plots()
    file_name = f"lda_plot.html"
    with open(file_name, "rb") as html_plot:
        html_text = html_plot.read()
        encoded_text = base64.b64encode(html_text).decode("utf-8")
        json_res["lda_plot"] = encoded_text
    os.system(f"rm {file_name}")

    return json_res