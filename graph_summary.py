

# %%
### importing the necessary libraries

from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
import numpy as np
import pandas
import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity

# %%
### Reading the data file

df = pandas.read_csv('Data/tennis_articles_v4.csv')
# df['article_text']

# %%
"""
## STEP 1 : Data Cleaning
### Cleaning sentences, by removing Non Alphabet Characters and converting to Lower Case Letters
"""

# %%
### cleaning sentences, by removing non alphabet characters and converting to lower case letters

s = ""
d = {}
for a in df['article_text']:
      s += a
# print s
sentences = sent_tokenize(s)
clean_sentences = []
for s in sentences:
    temp = re.sub("[^a-zA-Z]"," ",s)
    temp = temp.lower()
    clean_sentences.append(temp)
    d[temp] = s 
# print clean_sentences

# %%
"""
### Removing Stop Words
"""

# %%
### defined a functiom for removing stop words which are downloaded from NLTk's list of english stop words

stop_words = stopwords.words('english')
def rem_stop(s):
    var = ""
    words = nltk.word_tokenize(s)
    for w in words:
        if( w not in stop_words):
           var+=w+" "
    return var


# %%
### removed the stop words using the function defined above

dict = {}
clean = []
# print clean_sentences
for s in clean_sentences:
    temp = rem_stop(s)
    clean.append(temp)
    dict[temp] = d[s]
# print clean  

# %%
### loaded pre trained word2vec model from Gensim

from gensim.models import KeyedVectors
filename = 'Data/GoogleNewsvectorsnegative300.bin'
model = KeyedVectors.load_word2vec_format(filename, binary=True)

# %%
"""
## STEP 2 : Sentence Vector Generation
### Vector Representations are created using pre trained word2vec model from Gensim
"""

# %%
### creating vector representation of sentences after extracting word vectors

# print(model)
word_embeddings = {}
words = list(model.key_to_index.keys())
# print len(words)
for a in words:
    word_embeddings[a]=model[a]

# print len(word_embeddings)


sentence_vectors = []
for i in clean:
  if len(i) != 0:
    v = sum([word_embeddings.get(w, np.zeros((300,))) for w in i.split()])/(len(i.split())+0.001)
  else:
    v = np.zeros((300,))
  sentence_vectors.append(v)


# %%
"""
## STEP 3 : Graph Formation
### Graph is formed where sentences are the nodes and edges are formed using Cosine Similarity between the sentences
"""

# %%
### generating the final summary after producing the graph using networkx and applying pagerank algo

sentence_similarity_martix = np.zeros([len(sentences), len(sentences)])
for i in range(len(sentences)):
  for j in range(len(sentences)):
    if i != j:
      sentence_similarity_martix[i][j] = cosine_similarity(sentence_vectors[i].reshape(1,300), sentence_vectors[j].reshape(1,300))[0,0]

sentence_similarity_graph = nx.from_numpy_array(sentence_similarity_martix)
scores = nx.pagerank(sentence_similarity_graph)
ranked_sentence = sorted(((scores[i],s) for i,s in enumerate(clean)), reverse=True)    

# %%
"""
## STEP 4 : Summary Generation
### Summary is produced using PageRank algorithm and top 5 ranked sentences are printed
"""

# %%
### printing the top 5 sentences
# print ranked
for i in range(5):
     print (dict[ranked_sentence[i][1]])