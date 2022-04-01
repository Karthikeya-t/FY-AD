import streamlit as st
import tensorflow as tf
import keras
import cv2
# video_file = open('Shoplifting031_x264.mp4', 'rb')
# video_bytes = video_file.read()
import os
uploadedfile = st.file_uploader("Upload Video ",type=["mp4", "ogv", "m4v", "webm"])
try:
    if uploadedfile is not None:
        file_details = {"FileName":uploadedfile.name,"FileType":uploadedfile.type}
        st.write(file_details)
        with open(uploadedfile.name, "wb") as f:
            f.write(uploadedfile.getbuffer())
        st.success("Saved File")
        video_bytes = uploadedfile.read()
        st.video(video_bytes)
        file_name = uploadedfile.name
        print(file_name)
except:
    print("something went wrong in file upload")

import io
import base64
import os
import cv2
import matplotlib



fps_of_video = int(cv2.VideoCapture(file_name).get(cv2.CAP_PROP_FPS))
frames_of_video = int(cv2.VideoCapture(file_name).get(cv2.CAP_PROP_FRAME_COUNT))

print("fps",fps_of_video," frames_of_video :",frames_of_video )

vidcap = cv2.VideoCapture(file_name)

count = 0
success = True
while success:
  success,image = vidcap.read()
  if success:
    cv2.imwrite("img/frame%09d.jpg" % count, image)
    success,image = vidcap.read()
    count += 1
  else:
    break
print(count)

print(">>>>>>>>> making images from video is done <<<<<<")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import os

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import roc_curve, auc, roc_auc_score

import tensorflow as tf
from os import listdir
from os.path import isfile, join, isdir
from PIL import Image

import shelve
from tensorflow.keras.optimizers import Adam

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from keras.layers import Conv2DTranspose, ConvLSTM2D, BatchNormalization, TimeDistributed, Conv2D, LayerNormalization
from keras.models import Sequential, load_model

#################################################

import matplotlib.pyplot as plt
from tensorflow import keras
model = keras.models.load_model('weights/model_att/')

def get_single_test(SINGLE_TEST_PATH):
    all_frames_single_test = []
    for f in sorted(listdir(SINGLE_TEST_PATH)):
        img = Image.open(join(SINGLE_TEST_PATH, f)).resize((64, 64))
        img = np.array(img, dtype=np.float32) / 256.0
        all_frames_single_test.append(img)
    return all_frames_single_test


def evaluate(SINGLE_TEST_PATH):
    test = get_single_test(SINGLE_TEST_PATH)
    test = np.array(test)
    reconstructed_sequences = model.predict(test)
    sequences_reconstruction_cost = np.array(
        [np.linalg.norm(np.subtract(test[i], reconstructed_sequences[i])) for i in range(0, test.shape[0])])
    sa = (sequences_reconstruction_cost - np.min(sequences_reconstruction_cost)) / np.max(sequences_reconstruction_cost)
    sr = 1.0 - sa
    return sr


SINGLE_TEST_PATH = "img"
op = evaluate(SINGLE_TEST_PATH)

import plotly.express as px

fig = px.line(op)
st.plotly_chart(fig, use_container_width=True)
# fig.show()

abnormal_dict = {}
j = 1
path = "img/frame000000000.jpg"
for i in op:
    abnormal_dict[path] = i
    j = j + 1
    path = "img/frame%09d.jpg" % j

# abnormal_dict
import operator

# from collections import OrderedDict

dict1 = abnormal_dict
sorted_tuples = sorted(dict1.items(), key=operator.itemgetter(1))
print(sorted_tuples)

sorted_dict = {}
for k, v in sorted_tuples[0:15]:
    sorted_dict[k] = v

print(sorted_dict)
st.write(sorted_dict.keys())
############
l=list(sorted_dict.keys())
from itertools import cycle
cols = cycle(st.columns(3)) # st.columns here since it is out of beta at the time I'm writing this
for idx, filteredImage in enumerate(l):
    next(cols).image(filteredImage, width=150, caption=l[idx].split('/')[-1])
################
from tensorflow import keras
cls_model = keras.models.load_model('weights/model_efficient_5_ep.hdf5')
cls_model.summary()

CLASS_LABELS = ['Abuse','Arrest','Arson','Assault','Burglary','Explosion','Fighting','Shoplifting','RoadAccidents','Robbery','Shooting',"Normal",'Stealing','Vandalism']

from os import listdir
from tensorflow.keras.preprocessing import image
op=[]
for i in sorted_dict.keys() :
  img = image.load_img(i, target_size=(64, 64))
  img = image.img_to_array(img)
  img = np.expand_dims(img, 0)
  preds = cls_model.predict(img)
  pred = np.argmax(preds)
  op.append(CLASS_LABELS[pred])
  print(i," --> ",CLASS_LABELS[pred])
  st.write(i," --> ",CLASS_LABELS[pred])
st.write("________")
st.write(set(op))

