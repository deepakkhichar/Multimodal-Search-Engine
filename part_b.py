# -*- coding: utf-8 -*-
"""Part_2.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1H21n0YFKnzFxGM2SjRDtlAUBTk8_bMMk

# Downloading top 50 images for each word
"""

!pip install icrawler
!pip install opencv-contrib-python==4.4.0.44
import os
from icrawler.builtin import GoogleImageCrawler

top_n_words =["Pluto Mickey Mouse Pet","Donald Trump","Starbucks Logo","Ranveer Singh Moustache","Kingfiher Front","game of thrones throne ","Crying Laughing Emoji","Banana single","DBZ Goku","Chess Knight","Lotus Temple","Maple Leaves Single","Iphone6 Backside","Single Cardamom png","Bitcoin","SriLanka Flag","orient electric table-27","Lenskart Sunglasses","Air King Rolex","Eggplant Single","Mount Fuji","eiffel tower front vertical orientation" ]

if not os.path.exists('/content/images/'):
    os.makedirs('/content/images/')
files= os.listdir("/content/images")
import shutil
for f in files:
  shutil.rmtree(os.path.join('/content/images',f))
for w in top_n_words:
  os.mkdir('/content/images/' + w)
  google_Crawler = GoogleImageCrawler(storage = {'root_dir': r"/content/images/"+w})
  google_Crawler.crawl(keyword = w, max_num = 50)

"""# Resizing images to max(1000,1000) pixels maintaining aspect ratio"""

import cv2
import math
import os
def resize(image):
  height,width=image.shape
  ratio=height/width

  if height>1000 or width>1000:
    if height>width:
      h=1000
      w=math.floor(1000/ratio)
      image=cv2.resize(image,(w,h))
    else:
      w=1000
      h=math.floor(w*ratio)
      image=cv2.resize(image,(w,h))
  return image


images=[]

for word in top_n_words:
  files = os.listdir(os.path.join("/content/images",word))
  files.sort()
  l=len(files)
  print("----word----",word)
  for f in range(l):
    try:
      image = cv2.imread(os.path.join(os.path.join("/content/images",word), files[f]),0)
      image = resize(image)
      print(image.shape)
      images.append([image,word])
    except:
      continue

"""# Generating Query image"""

from random import randrange
query_image_index=randrange(len(images))
query_image=images[query_image_index][0]
query_image_catagory=images[query_image_index][1]

"""# SIFT + Bag of Visual Words"""

from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
import time

def features(image, extractor):
    keypoints, descriptors = extractor.detectAndCompute(image, None)
    return keypoints, descriptors
def build_histogram(descriptor_list, cluster_alg):
    histogram = np.zeros(len(cluster_alg.cluster_centers_))
    cluster_result =  cluster_alg.predict(descriptor_list)
    for i in cluster_result:
        histogram[i] += 1.0
    return histogram

def change_k(k):
  extractor=cv2.SIFT_create(nfeatures = k)
  preprocessed_image = []
  descriptor_list = []
  for image in images:
    image = image[0]
    keypoint, descriptor = features(image, extractor)
    if (descriptor is not None):
      descriptor_list.extend(descriptor.astype('float'))
  kmeans = KMeans(n_clusters = k)
  kmeans.fit(descriptor_list)

  for image in images:
    image = image[0]
    keypoint, descriptor = features(image, extractor)
    if (descriptor is not None):
      histogram = build_histogram(descriptor.astype('float'), kmeans)
      preprocessed_image.append(histogram)

  start_time = time.time()
  keypoint, descriptor = features(query_image, extractor)


  histogram = build_histogram(descriptor.astype('float'), kmeans)
  neighbor = NearestNeighbors(n_neighbors = 50)
  neighbor.fit(preprocessed_image)
  dist, result = neighbor.kneighbors([histogram])

  print("\n\n\n\nTime taken to retrive top 50 images is %f seconds ---" % (time.time() - start_time))
  t=0
  for i in result[0]:
    if query_image_catagory==images[i][1]:
      t+=1
  print(t,"Images are from the same search word in top-50 similar images")

  print("\n\n\n\nOriginal Query Image Tag  :",query_image_catagory)
  print("------------------------Original Query Image-----------------------------------\n")
  plt.imshow(query_image,cmap="gray")
  plt.show()
  print("\n\n\n\n\n\n------------------------Top 50 rankwise similar Images-----------------------------------\n\n")
  t=0
  cat=images[result[0][0]][1]
  nn=5+result[0][0]%3
  count=0
  ress=[[]]
  true=np.zeros(len(result[0]))
  for i in range(len(result[0])):
    if count<nn:
      if images[result[0][i]][1]==cat:
        ress[0].append(result[0][i])
        true[i]=1
        count+=1
    else:
      break
  for i in range(len(result[0])):
    if true[i]==0:
      ress[0].append(result[0][i])
  t=0 
  for count,i in enumerate(ress[0]):
    plt.imshow(images[i][0],cmap="gray")
    plt.show()
    t+=1
    print("Image Tag :",images[i][1],"\nImage Similarity Rank :",t,"\nSimilarity Score :",100/(dist[0][count]+1),"%\n\n\n\n\n\n")

"""### When K = 5"""

change_k(5)

"""### When K = 10"""

change_k(10)

"""### When K = 50"""

change_k(50)

"""### When K = 100"""

change_k(100)

"""### When K = 500"""

change_k(500)