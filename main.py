# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import cv2
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
width=1
height=1
rows = 5
cols = 3
axes=[]
fig=plt.figure(figsize=(17,15))
ab=['img/frame000000203.jpg','img/frame000000202.jpg','img/frame000000201.jpg']
l=sorted(ab) # l=sorted_dict.keys()

from itertools import cycle
cols = cycle(st.columns(4)) # st.columns here since it is out of beta at the time I'm writing this
for idx, filteredImage in enumerate(l):
    next(cols).image(filteredImage, width=150, caption=l[idx].split('/')[-1])


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
