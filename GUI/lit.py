#Importing Libraries
import streamlit as st
import wikipediaapi
import nltk
import re
import warnings
import gensim
import pickle
import time
import pandas as pd
import os
import cv2
import sklearn
import pickle
import numpy
import numpy as np
import matplotlib.pyplot as plt
import os
import math
import speech_recognition as sr

from nltk.stem import PorterStemmer
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from gensim.models import Word2Vec, KeyedVectors   
from IPython.display import HTML
from PIL import Image
from icrawler.builtin import GoogleImageCrawler
from joblib import dump, load
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from itertools import cycle

nltk.download('stopwords')
nltk.download('stopwords')
nltk.download('punkt')

warnings.filterwarnings(action = 'ignore')


# Title
st.title("Yo! Welcome to the SOTA search!!")
# Header
# st.header("ELL 786")
# Subheader
st.subheader("Multimodal Search Engine")


wiki_wiki = wikipediaapi.Wikipedia('en')


#Loading Text Search Models
with open('corpus_learning_file.pickle','rb') as corpus_learning_file:
        [english_stopwords,word_freq,N_phrases,word_dictionary,cleaned_text_dict,doc_vectors,phrases_list] = pickle.load(corpus_learning_file)

try:
    Skip_Gram_model2 = Word2Vec.load("Skip_Gram_model2.model")
except:
    print("Error loading")

#Text Functions
def TF_IDF_search(query,english_stopwords,word_freq,N_phrases,word_dictionary):
  url_prefix = "https://en.wikipedia.org/wiki/"
  ps = PorterStemmer()
  search_word = query
  query = ps.stem(query)
  if query in english_stopwords:
    print("Please give a more specific query for ", search_word)
  if query not in word_freq:
    print("Sorry! No good search available for ",search_word)
  else:
    Query_result_list = []
    IDF = math.log2(N_phrases/word_freq[query])
    sorted_doc_list = sorted( word_dictionary[query] , key = lambda x: x[1], reverse = True)
    print("Results for your query ", search_word ," are:")
    for rank, result in enumerate(sorted_doc_list):
      score = result[1]
      Query_result_list.append([rank+1,score, result[0], url_prefix+result[0].replace(" ","_")])

    Header = ["Rank","TF-IDF Score","Article Heading","Link"]

    return Query_result_list

def Word2Vec_search(query,english_stopwords,Skip_Gram_model2,cleaned_text_dict,doc_vectors,phrases_list):
  url_prefix = "https://en.wikipedia.org/wiki/"
  ps = PorterStemmer()

  search_word = query
  query = ps.stem(query.lower())
  if query in english_stopwords:
    print("Please give a more specific query for ", search_word)
  else:
    Query_result_list = []
    query_vector = Skip_Gram_model2.wv[query]
    similar_words=  Skip_Gram_model2.wv.most_similar(positive = [query_vector], topn=10)
    print( " You may also like to search:  " )
    for similar_term in similar_words:
      print("-----> ",similar_term)
    for phrase in phrases_list:
      if phrase not in cleaned_text_dict or len(cleaned_text_dict[phrase])==0:
        continue
      document_vector = doc_vectors[phrase]
      cosine_similarity = numpy.dot(document_vector, query_vector)/(numpy.linalg.norm(query_vector)* numpy.linalg.norm(document_vector))
      Query_result_list.append([cosine_similarity, phrase,url_prefix+phrase.replace(" ","_")])
      
    Query_result_list = sorted(Query_result_list , key = lambda x: x[0], reverse =True)
    print("Top 50 results for your query ", search_word ," are:")
    for rank, result in enumerate(Query_result_list):
      Query_result_list[rank] = [rank+1] + Query_result_list[rank]

    Header = ["Rank","Similarity Score","Article Heading","Link"]
    return Query_result_list[0:50]

def Text_search(query, algo):
  if algo == 0:
    return TF_IDF_search(query,english_stopwords,word_freq,N_phrases,word_dictionary)
  else:
    return Word2Vec_search(query,english_stopwords,Skip_Gram_model2,cleaned_text_dict,doc_vectors,phrases_list)

def convert(row):
    return '<a href="{}">{}</a>'.format(row['URL'],  row.name+1)

def Word2Vec_search_audio(query,english_stopwords,Skip_Gram_model2,cleaned_text_dict,doc_vectors,phrases_list):
  url_prefix = "https://en.wikipedia.org/wiki/"
  ps = PorterStemmer()

  search_word = query
  query = query.split(" ")
  query = list(map(lambda x: ps.stem(x.lower().strip()),query))
  query = list(filter(lambda x: x not in english_stopwords,query))
  if len(query)==0:
    print("Please give a more specific query for ", search_word)
  else:
    Query_result_list = []
    query_vector = sum(list(map(lambda x: Skip_Gram_model2.wv[x], query)))/len(query)
    similar_words=  Skip_Gram_model2.wv.most_similar(positive = [query_vector], topn=10)
    print( " You may also like to search:  " )
    for similar_term in similar_words:
      print("-----> ",similar_term)
    for phrase in phrases_list:
      if phrase not in cleaned_text_dict or len(cleaned_text_dict[phrase])==0:
        continue
      document_vector = doc_vectors[phrase]
      cosine_similarity = numpy.dot(document_vector, query_vector)/(numpy.linalg.norm(query_vector)* numpy.linalg.norm(document_vector))
      Query_result_list.append([cosine_similarity, phrase,url_prefix+phrase.replace(" ","_")])
      
    Query_result_list = sorted(Query_result_list , key = lambda x: x[0], reverse =True)
    print("Top 50 results for your query ", search_word ," are:")
    for rank, result in enumerate(Query_result_list):
      Query_result_list[rank] = [rank+1] + Query_result_list[rank]

    Header = ["Rank","Similarity Score","Article Heading","Link"]
    return Query_result_list[0:50]




#Loading Image Prediction Models
kmeans=load('kmeans.joblib')
with open('parrot.pickle', 'rb') as f:
  [preprocessed_image, images] = pickle.load(f)

#Image Functions
def features(image, extractor):
    keypoints, descriptors = extractor.detectAndCompute(image, None)
    return keypoints, descriptors

def build_histogram(descriptor_list, cluster_alg):
    histogram = np.zeros(len(cluster_alg.cluster_centers_))
    cluster_result =  cluster_alg.predict(descriptor_list)
    for i in cluster_result:
        histogram[i] += 1.0
    return histogram

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

extractor=cv2.SIFT_create(nfeatures = 50)

def query(query_image):
    query_image=resize(query_image)
    start_time = time.time()
    keypoint, descriptor = features(query_image, extractor)


    histogram = build_histogram(descriptor.astype('float'), kmeans)
    neighbor = NearestNeighbors(n_neighbors = 50)
    neighbor.fit(preprocessed_image)
    dist, result = neighbor.kneighbors([histogram])

    #   print("\n\n\n\nTime taken to retrive top 50 images is %f seconds ---" % (time.time() - start_time))
    t=0
    query_image_catagory=images[result[0][0]][1]
    for i in result[0]:
        if query_image_catagory==images[i][1]:
            t+=1
    tt= t
    # st.write("{} Images are from the same search word in top-50 similar images".format(t))

    #   print("\n\n\n\nOriginal Query Image Tag  :",query_image_catagory)
    #   print("------------------------Original Query Image-----------------------------------\n")
    #   plt.imshow(query_image,cmap="gray")
    #   plt.show()
    #   print("\n\n\n\n\n\n------------------------Top 50 rankwise similar Images-----------------------------------\n\n")
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
    matrix=[]
    tag_matrix = []
    rank_matrix = []
    similarity_matrix = []
    image_matrix= []
    
    for count,i in enumerate(ress[0]):
        
        similarity_matrix.append(100/(dist[0][count]+1))
        tag_matrix.append(images[i][1])
        rank_matrix.append(t+1)
        image_matrix.append(images[i][0])
        
        t+=1
        
        #plt.imshow(images[i][0],cmap="gray")
        #plt.show()
        #matrix.append([images[i][1],t,100/(dist[0][count]+1),images[i][0]])
        #print("Image Tag :",images[i][1],"\nImage Similarity Rank :",t,"\nSimilarity Score :",100/(dist[0][count]+1),"%\n\n\n\n\n\n")
    
    
    matrix = [tag_matrix, rank_matrix, similarity_matrix, image_matrix]
    return matrix,tt



# radio button
# first argument - title of the radio button
# second argument - options for the ratio button
status = st.radio("Choose Query Algorithm", ('Text - Tf-Idf', 'Text - Word2Vec','Image','Audio','Video'))


if (status == 'Text - Tf-Idf'): 

    name = st.text_input("Enter Query", "Tf-Idf results awaiting!")

    if(st.button('Submit')):

        query = name.title()

        starttime = time.time()
        matrix = pd.DataFrame(Text_search(query, 0))
        endtime = time.time()
        time_taken = ("Time taken for TF-IDF search : {} sec").format('%.5f' % (endtime-starttime))
        st.write(time_taken)

        matrix.columns = ["Rank", "Similarity Score", "Heading", "URL"]
        matrix['URL'] = matrix.apply(convert, axis=1)        
        
        st.write(matrix.to_html(escape=False, index=False), unsafe_allow_html=True)
        st.success("Success")


if(status == 'Text - Word2Vec'):

  name = st.text_input("Enter Query", "Word2Vec results on the way!")
  
  if(st.button('Submit')):

      query = name.title()

      starttime = time.time()
      matrix = pd.DataFrame(Text_search(query, 1))
      endtime = time.time()
      time_taken = ("Time taken for TF-IDF search : {} sec").format('%.5f' % (endtime-starttime))
      st.write(time_taken)

      matrix.columns = ["Rank", "Similarity Score", "Heading", "URL"]
      matrix['URL'] = matrix.apply(convert, axis=1)         

      st.write(matrix.to_html(escape=False, index=False), unsafe_allow_html=True)
      st.success("Success")


if(status == 'Image'):

    def load_image(image_file):
      img = Image.open(image_file)
      return img

    def save_uploadedfile(uploadedfile):  
      with open("Saved_File.png","wb") as f:
        f.write(uploadedfile.getbuffer())

    st.subheader("Image")  
    image_file = st.file_uploader("Upload Images", type=["png","jpg","jpeg"])

    if image_file is not None:

        # To View Uploaded Image
        img = load_image(image_file) 
        # Saving the uploaded image  
        save_uploadedfile(image_file)        
        # Reading the uploaded image
        read_file = cv2.imread("Saved_File.png",0)

        # Running the Algorithm on query
        matrix, tt = query(read_file)

        st.write("")
        st.write("")
        st.write("\n\n\n\nOriginal Query Image Tag  :",matrix[0][0])
        st.write("")
        st.write("")
        st.write("--------------------------Original Query Image--------------------------\n")
        st.image(load_image(image_file),width=250)
        st.write("\n\n\n\n\n\n--------------------Top 50 rankwise similar Images--------------------\n\n")
        st.write("")
        st.write("")
        st.write("{} Images are from the same search word in top-50 similar images".format(tt))
        st.write("")
        st.write("")

        #Truncating similarity score
        matrix[2] = [ '%.4f' % elem for elem in matrix[2] ]


        filteredImages = matrix[3] # your images here
        caption = matrix[0]
        idx = 0 
        for _ in range(len(filteredImages)-1): 
            cols = st.columns(3) 
            
            if idx < len(filteredImages): 
                cols[0].image(filteredImages[idx], width=150, caption="Rank = {} \n".format(idx+1)+"Similarity Score = {} \n".format(matrix[2][idx])+caption[idx])
            idx+=1
            
            if idx < len(filteredImages):
                cols[1].image(filteredImages[idx], width=150, caption="Rank = {} \n".format(idx+1)+"Similarity Score = {} \n".format(matrix[2][idx])+caption[idx])
            idx+=1

            if idx < len(filteredImages): 
                cols[2].image(filteredImages[idx], width=150, caption="Rank = {} \n".format(idx+1)+"Similarity Score = {} \n".format(matrix[2][idx])+caption[idx])
                idx = idx + 1
            else:
                break


if(status == 'Audio'):
  
  path = "C:\\Users\\AYAN\\ass3\\"
  audio_file = open((path+'final_audio.wav'), 'rb')
  audio_bytes = audio_file.read()

  st.audio(audio_bytes, format='audio/ogg')

  # This will take voice input as WAV format .

  # initialize the recognizer
  r = sr.Recognizer()

  path = "C:\\Users\\AYAN\\ass3\\final_audio.wav"

  # open the file
  retrived_text = ""
  with sr.AudioFile(path) as source:
      # listen for the data (load audio to memory)
      audio_data = r.record(source)
      # recognize (convert from speech to text)
      text = r.recognize_google(audio_data)
      retrived_text = text.replace("'s","");
      #print(text)

  st.write(retrived_text)

  starttime = time.time()
  matrix = pd.DataFrame(Word2Vec_search_audio(retrived_text,english_stopwords,Skip_Gram_model2,cleaned_text_dict,doc_vectors,phrases_list))
  endtime = time.time()
  time_taken = ("Time taken for Word2Vec_Audio search : {} sec").format('%.5f' % (endtime-starttime))
  st.write(time_taken)

  matrix.columns = ["Rank", "Similarity Score", "Heading", "URL"]
  matrix['URL'] = matrix.apply(convert, axis=1)         

  st.write(matrix.to_html(escape=False, index=False), unsafe_allow_html=True)
  st.success("Success")



if(status == 'Video'):
    pass



















