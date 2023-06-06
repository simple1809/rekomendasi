import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt

import seaborn as sns
import plotly.express as px

from sklearn.feature_extraction.text import TfidfVectorizer
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from sklearn.metrics.pairwise import cosine_similarity

from zipfile import ZipFile
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from pathlib import Path

tv = TfidfVectorizer(max_features=5000)
stem = StemmerFactory().create_stemmer()
stopword = StopWordRemoverFactory().create_stop_word_remover()

data_tourism_rating = pd.read_csv('tourism_rating.csv')
data_tourism_with_id = pd.read_csv('tourism_with_id.csv')
data_user = pd.read_csv('user.csv')

print(data_tourism_rating.head())
print(data_tourism_with_id.head())
print(data_user.head())

data_tourism_with_id.drop(['Rating','Time_Minutes','Coordinate','Lat','Long','Unnamed: 11','Unnamed: 12'],axis=1,inplace=True)

print("-----")
print(data_tourism_with_id)

data_rekomendasi = pd.merge(data_tourism_rating.groupby('Place_Id')['Place_Ratings'].mean(),data_tourism_with_id,on='Place_Id')
print("------")
print(data_rekomendasi)

def preprocessing(data):
    data = data.lower()
    data = stem.stem(data)
    data = stopword.remove(data)
    return data

data_content_based_filtering = data_rekomendasi.copy()
data_content_based_filtering['Tags'] = data_content_based_filtering['Description'] + ' ' + data_content_based_filtering['Category']
data_content_based_filtering.drop(['Price','Place_Ratings','Description','Category'],axis=1,inplace=True)
print("----")
print("Data Filltering")
print(data_content_based_filtering)

data_content_based_filtering.Tags = data_content_based_filtering.Tags.apply(preprocessing)
print("----")
print(data_content_based_filtering)

vectors = tv.fit_transform(data_content_based_filtering.Tags).toarray()
print("-----")
print("Vectors")
print(vectors)

similarity = cosine_similarity(vectors)
similarity[0][1:10]
print("----")
print("Similarity")
print(similarity)

def recommend_by_content_based_filtering(nama_tempat):
    nama_tempat_index = data_content_based_filtering[data_content_based_filtering['City']==nama_tempat].index[0]
    distancess = similarity[nama_tempat_index]
    nama_tempat_list = sorted(list(enumerate(distancess)),key=lambda x: x[1],reverse=True)[1:20]
    
    recommended_nama_tempats = []
    for i in nama_tempat_list:
        recommended_nama_tempats.append(([data_content_based_filtering.iloc[i[0]].Place_Name]+[i[1]]))
        
    return recommended_nama_tempats
print("----")
print("Berikut Hasil Rekomendasi :")
print("^^")
print(recommend_by_content_based_filtering('Jakarta'))





