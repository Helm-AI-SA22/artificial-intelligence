import requests
import wget
import pandas as pd
import os
import base64
import json
import time
import matplotlib.pyplot as plt
import imgkit

from htmlwebshot import WebShot
shot = WebShot()
shot.quality = 100

image = shot.create_pic(html="ldaPlot.html")