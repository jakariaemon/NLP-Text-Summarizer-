# %%
"""
## Generating Summary using tf-idf Algorithm 
### STEP 1 : Data cleaning ( removing non letter characters, turning to lower case letters )
### STEP 2 : Getting tf-idf score of sentences
### STEP 3 : Summary Generation
"""

# %%
"""
## Initial Phase
### Importing Libraries and Reading Data 
"""

# %%
### importing the necessary libraries

from nltk.corpus import stopwords
import numpy as np
import pandas
import nltk
import re
# from __future__ import division

# %%
df = pandas.read_csv('Downloads/tennis_articles_v4.csv')

# %%
"""
### Tokenizing sentences into words which would be used for calculating tf-idf scores
"""

# %%
### tokenized the sentences from the different news articles

from nltk.tokenize import sent_tokenize
s = ""
for a in df['article_text']:
      s += a
sentences = sent_tokenize(s)
# sentences

# %%
"""
## STEP 1 : Data Cleaning
### Cleaning sentences, by removing Non Alphabet Characters and converting to Lower Case Letters
"""

# %%
### pre processes the sentences by removing non alphabet characters and converting them to lower case letters 
### and stored in variable text

dict = {}
text=""
for a in sentences:
    temp = re.sub("[^a-zA-Z]"," ",a)
    temp = temp.lower()
    dict[temp] = a
    text+=temp
# print text

# %%
"""
## STEP 2 : Getting tf-idf score of sentences
### Finding term frequency ( tf ) of words found in text
"""

# %%
### calculated the frequency of the words found in text

stopwords = nltk.corpus.stopwords.words('english')
word_frequencies = {}
for word in nltk.word_tokenize(text):
    if word not in stopwords:
        if word not in word_frequencies.keys():
            word_frequencies[word] = 1
        else:
            word_frequencies[word] += 1
# print len(word_frequencies)

# %%
"""
### Finding weighted frequency of the words
"""

# %%
### finding weighted frequency of the words

max_freq = max(word_frequencies.values())

for w in word_frequencies :
      word_frequencies[w]/=max_freq
# print word_frequencies

# %%
"""
### Calculating sentence scores from the word frequncies
"""

# %%
### calculating sentence scores from the word frequncies

sentence_scores = {}
for sent in sentences:
    for word in nltk.word_tokenize(sent.lower()):
        if word in word_frequencies.keys():
            if len(sent.split(' ')) < 30:
                if sent not in sentence_scores.keys():
                    sentence_scores[sent] = word_frequencies[word]
                else:
                    sentence_scores[sent] += word_frequencies[word]

# %%
"""
## STEP 3 : Summary Generation
### Outputting the top 17 sentences as the summary
"""

# %%
### getting the summary by taking top score sentences

import heapq
summary_sentences = heapq.nlargest(17, sentence_scores, key=sentence_scores.get)
summary = ' '.join(summary_sentences)
print(summary)