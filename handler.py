import os
import base64
import warnings
warnings.filterwarnings("ignore")

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
        id, prob = doc_info

        document = {
            "id": id,
            "topics": []
        }

        if prob.sum() > 0:

            for topic in range(len(prob)):
                topic_info = {
                    "id": topic,
                    "affinity": prob[topic]
                }

                document["topics"].append(topic_info)
        
        document["topics"].append({
            "id": -1,
            "affinity": 1 - prob.sum()
        })

        json_res["documents"].append(document)


    json_res["topics"] = []

    for i in range(len(names)):
        topic_name = names[i]
        topic = {
            "id": i,
            "name": topic_name
        }
        json_res["topics"].append(topic)

    json_res["topics"].append({
        "id": -1,
        "name": "noise"
    })

    print("Generating the plots")
    plots = model.get_plots()
    json_res["topicsVisualization"] = {}

    # for plot_name, plot in plots.items():
        
    #     interactive_plot_name = f"{plot_name}Interactive"
    #     static_plot_name = f"{plot_name}Static"


    #     if plot == None:
    #         json_res["topicsVisualization"][interactive_plot_name] = None
    #         json_res["topicsVisualization"][static_plot_name] = None
    #         continue

    #     # generate html_plot
    #     file_name = f"{interactive_plot_name}.html"
    #     plot.write_html(file_name)
    #     with open(file_name, "rb") as html_plot:
    #         html_text = html_plot.read()
    #         encoded_text = base64.b64encode(html_text).decode("utf-8")
    #         json_res["topicsVisualization"][interactive_plot_name] = encoded_text

    #     # generate png image
    #     image_byte = plot.to_image(format="png")
    #     encoded_image = base64.b64encode(image_byte).decode("utf-8")
    #     json_res["topicsVisualization"][static_plot_name] = encoded_image
        
    #     os.system(f"rm {file_name}")

    for plot_name, plot in plots.items():
        
        if plot == None:
            json_res["topicsVisualization"][plot_name] = None
            continue

        file_name = f"{plot_name}.html"
        plot.write_html(file_name)
        with open(file_name, "rb") as html_plot:
            html_text = html_plot.read()
            encoded_text = base64.b64encode(html_text).decode("utf-8")
            json_res["topicsVisualization"][plot_name] = encoded_text
            
        os.system(f"rm {file_name}")



    return json_res




def fast_api_handler(json_req, model, num_topics=None):

    json_res = {}

    ids = []
    texts = []
    for doc in json_req["documents"]:
        ids.append(doc["id"])
        texts.append(doc["abstract"])


    print("Started topic modeling with LDA")
    if num_topics is None:
        probs, names = model.train(texts)
    else:
        probs, names = model.train(texts, num_topics)

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


    json_res["topicsVisualization"] = {}
    model.get_plots()
    file_name = f"ldaPlot.html"
    with open(file_name, "rb") as html_plot:
        html_text = html_plot.read()
        encoded_text = base64.b64encode(html_text).decode("utf-8")
        json_res["topicsVisualization"]["ldaPlot"] = encoded_text
    os.system(f"rm {file_name}")

    return json_res