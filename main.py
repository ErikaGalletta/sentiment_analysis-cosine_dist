import pandas as pd
import json
# Initialize your API key
import os
from openai import OpenAI
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial import distance

from sklearn.manifold import TSNE
import numpy as np


class search_reviews:
    def __init__(self):
        self.client=OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self.embeddings=self.data_processing()
    def find_n_closest(self,query_vector, n): #when calling the function, a number can be specified if we want a different number of closest matches
        distances=[]
        for index, embedding in enumerate(self.embeddings):
            dist=distance.cosine(query_vector, embedding)
            distances.append({'distance':dist,'index':index})
    #3- extract the texts with the smallest cosine distance
        distances_sorted=sorted(distances, key=lambda x: x['distance'])
        return distances_sorted[0:n]

    def return_closest_doc(self,embedded_list,original_list):
        result={}
        
        for inputs in embedded_list.data:
            result[original_list[inputs.index]]=[]
            index=self.find_n_closest(inputs.embedding, n=3 )
            for i in index:
                result[original_list[inputs.index]].append(self.json_objects[i['index']])
        return result
    def data_processing(self):
        self.json_objects = []
        reviews = pd.read_csv("womens_clothing_e-commerce_reviews.csv")
        # Iterate through each row of the dataframe
        for index, row in reviews.iterrows():
            # Create a dictionary where column names are keys and row values are values
            row_dict = row.to_dict()

            # Convert the dictionary to a JSON object (optional step)
            json_object= json.dumps(row_dict)

            # Add to list of JSON objects
            self.json_objects.append(json_object)
        response= self.embedd(self.json_objects)
        return [inputs.embedding for inputs in response.data]
#the function returns the indexes of the 3 closest items from the embedded list
#to visualize the results:
    def embedd(self,text):
        return self.client.embeddings.create(
    model="text-embedding-3-small",input=text)
    
    def plot_vectors(self):
        tsne=TSNE(n_components=2, #resulting number of dimensions
             )
        embeddings_2d=tsne.fit_transform(np.array(self.embeddings))
        plt.scatter(embeddings_2d[:,0], embeddings_2d[:,1])


Clothes_review=search_reviews()
Clothes_review.plot_vectors()

    
categories_query = ['quality', 'fit', 'style', 'comfort']
categories_embeddings=Clothes_review.embedd(categories_query)
response=Clothes_review.return_closest_doc(categories_embeddings,categories_query)
for ind in response.keys():
    
    print( ind, ":\n")
    for v in response[ind]:
        print(v)

  query_review=["Absolutely wonderful - silky and sexy and comfortable"]
query_embeddings=Clothes_review.embedd(query_review)

response=Clothes_review.return_closest_doc(query_embeddings,query_review)
most_similar_reviews=[]
for ind in response.keys():
    
    for v in response[ind]:
        most_similar_reviews.append(v)
print(f'most similar review: {most_similar_reviews[0]}')
