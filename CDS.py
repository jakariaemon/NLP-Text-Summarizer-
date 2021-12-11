# %%
"""
# Text Summarization using Connected Dominating Set 

### STEP 1 : Data cleaning ( removing stop words, non letter characters, turning to lower case letters )
### STEP 2 : Sentence vector representation
### STEP 3 : Graph formation where edges formed using cosine similarity between sentences
### STEP 4 : Finding minimum Connected Dominating Set and outputting the summary 
"""

# %%
"""
## Initial Phase
### Importing Libraries and Reading Data 
"""

# %%
import matplotlib.pyplot as plt
import networkx as nx
import networkx.algorithms.approximation as nxaa
from collections import OrderedDict, deque
import copy
import operator

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
### reading data file

df = pandas.read_csv('Data/tennis_articles_v4.csv')

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

# %%
"""
## STEP 4 : Finding minimum Connected Dominating Set and Outputting the summary
### minimum Connected Dominating Set is found using a Greedy Approach which can be summarized in the following 3 steps :
### 1. Initialization : 
### Take the node with maximum degree as the starting node
### Enqueue the neighbor nodes of starting node to Q in descending order by their degree
### Maintain a priority queue centrally to decide whether an element would be a part of CDS.
### 2. CDS Calculation :
### Check if the graph after removing u is still connected
### Add neighbors of u to the priority queue, which never are inserted into Q
### 3. Result Verification :
### Verify the set is Dominating and Connected
### Output the Result
"""

# %%
"""
### 1. Initialization
"""

# %%
G = nx.Graph()
assert nx.is_connected(G)

### finding minimum connected dominating set using a greedy approach

G2 = copy.deepcopy(G)

# Step 1: initialization
# take the node with maximum degree as the starting node
starting_node = max(dict(G2.degree()).items(), key=operator.itemgetter(1))[0] 
fixed_nodes = {starting_node}

# Enqueue the neighbor nodes of starting node to Q in descending order by their degree
neighbor_nodes = G2.neighbors(starting_node)
neighbor_nodes_sorted =list( OrderedDict(sorted(dict(G2.degree(neighbor_nodes)).items(), key=operator.itemgetter(1), reverse=True)).keys())

priority_queue = deque(neighbor_nodes_sorted) # a priority queue is maintained centrally to decide whether an element would be a part of CDS.
# print([starting_node]+neighbor_nodes_sorted)
inserted_set = set(neighbor_nodes_sorted + [starting_node])


# %%
"""
### 2. CDS Calculation
"""

# %%
# Step 2: calculate the cds
while priority_queue:
    u = priority_queue.pop()

# check if the graph after removing u is still connected
rest_graph = copy.deepcopy(G2)
rest_graph.remove_node(u)

if nx.is_connected(rest_graph):
    G2.remove_node(u)
else: # is not connected 
    fixed_nodes.add(u)

# add neighbors of u to the priority queue, which never are inserted into Q
inserted_neighbors = set(G2.neighbors(u)) - inserted_set
inserted_neighbors_sorted = OrderedDict(sorted(dict(G2.degree(inserted_neighbors)).items(),key=operator.itemgetter(1), reverse=True)).keys()

priority_queue.extend(inserted_neighbors_sorted)
inserted_set.update(inserted_neighbors_sorted)


# %%
"""
### 3. Result Verification
"""

# %%
# Step 3: verify the result
assert nx.is_dominating_set(G, fixed_nodes) and nx.is_connected(G.subgraph(fixed_nodes))


# %%
"""
### Outputting the set formed in the previous step as the summary
"""

# %%
print (fixed_nodes)