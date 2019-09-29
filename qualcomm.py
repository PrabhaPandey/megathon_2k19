import re
import os
import nltk
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
import csv
import numpy
import sys
from numpy import linalg as LA

docs = []
content = sys.argv[1]
csv_file = open(content, 'rb')
for line in csv_file.readlines():
	# print(line)
	page_content = line.decode().split(',')[1]
	tokenizer = RegexpTokenizer(r'[a-zA-Z]+')
	tokens = tokenizer.tokenize(page_content)
	tokens = [x for x in tokens]
	stemmer = SnowballStemmer('english')
	words = []
	for x in tokens:
		words.append(stemmer.stem(x))
	page_content = ' '.join(words)
	docs.append(page_content)


abstract = []
abs = sys.argv[2]
csv_file = open(abs, 'rb')
for line in csv_file.readlines():
	# print(line)
	page_content = line.decode().split(',')[1]
	tokenizer = RegexpTokenizer(r'[a-zA-Z]+')
	tokens = tokenizer.tokenize(page_content)
	tokens = [x for x in tokens]
	stemmer = SnowballStemmer('english')
	words = []
	for x in tokens:
		words.append(stemmer.stem(x))
	page_content = ' '.join(words)
	abstract.append(page_content)


import re, math
from collections import Counter

WORD = re.compile(r'\w+')

def get_cosine(vec1, vec2):
     intersection = set(vec1.keys()) & set(vec2.keys())
     numerator = sum([vec1[x] * vec2[x] for x in intersection])

     sum1 = sum([vec1[x]**2 for x in vec1.keys()])
     sum2 = sum([vec2[x]**2 for x in vec2.keys()])
     denominator = math.sqrt(sum1) * math.sqrt(sum2)

     if not denominator:
        return 0.0
     else:
        return float(numerator) / denominator

def text_to_vector(text):
     words = WORD.findall(text)
     return Counter(words)

result_matrix = []
for d in docs:
	cos_for_one_doc = []
	for a in abstract:
		vec1 = text_to_vector(d)
		vec2 = text_to_vector(a)
		cosine = get_cosine(vec1,vec2)
		cos_for_one_doc.append(cosine)
	result_matrix.append(cos_for_one_doc)

result_matrix = numpy.array(result_matrix).T.tolist()

a = numpy.asarray(result_matrix)
numpy.savetxt("similarity_matrix.csv", a, delimiter=",")