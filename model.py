# filtering and categorizing words
import json
import time
import os, sys
import asyncio
import pickle # convert python model to binary code
from functools import lru_cache
from datetime import datetime

import pandas as pd
import numpy as np

# loading the model in pickle files 
model = pickle.load(open('pickle_files/text_classifier.pkl', 'rb')) # reding the numeric representation of model from the pickle file
tfidf_vectorizor = pickle.load(open('pickle_files/text_vectorizer.pkl', 'rb')) # reding from the TF-IDF sklearn vectorizor (Term Frequency Invere Document Frequency)
label_encoder = pickle.load(open('pickle_files/text_encoder.pkl', 'rb')) # encoded labels 


# training the model
def processing(inPath, outPath):

	# reading csv files
	input_df = pd.read_csv(inPath)
	# vectorize the data
	features = tfidf_vectorizor.transform(input_df['body']) # specifying the column you want to vectorize

	# predicting the classes
	prediction = model.predict(features)
	# converting output labels to category
	input_df['category'] = label_encoder.inverse_transform(prediction)

	# saving results to a different file
	output_df = input_df[['id', 'category']]
	output_df.to_csv(outPath, index=False)

# processing('CSV_files/test_set.csv', 'response.csv')

# vie the data
def data(newData):

	data_file = pd.read_csv(newData)
	df = pd.DataFrame(data_file)

	return data_file['category'].count()

if __name__ == '__main__':
	print(data('response.csv'))