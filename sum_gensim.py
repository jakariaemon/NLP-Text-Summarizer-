# %%
"""
## Summary Generation using Gensim's Summarize method
### Data is read in the initial phase and given as input to Gensim's summarize method. Summaries obtained by varying different input parameter values are also obtained. 
"""

# %%
"""
## Initial Phase
### Importing Libraries and Reading Data 
"""

# %%
import gensim
import logging
import numpy
import pandas
from gensim.summarization import summarize, keywords
import re

# %%
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# %%
df = pandas.read_csv('Downloads/tennis_articles_v4.csv')

# %%
"""
### Storing data from input file and replacing ' with white space 
"""

# %%
sentences = ""
for a in df['article_text']:
    sentences+=a
sentences = re.sub("'","",sentences)
sentences

# %%
"""
## Results
### Result Obtained from Gensim Summary
"""

# %%
print 'summary:'
print summarize(sentences)

# %%
"""
### Shorter Summary obtained by setting ratio to 0.1
"""

# %%
print 'summary:'
print summarize(sentences, ratio = 0.1)

# %%
"""
### Summary obtained as a complete paragraph with ratio as 0.1 
"""

# %%
print 'summary:'
print summarize(sentences, ratio = 0.1, split = True)

# %%
"""
### A more concise summary with ratio as 0.01
"""

# %%
print 'summary:'
print summarize(sentences, ratio = 0.01, split = True)

# %%
"""
### Different keywords identified by Gensim while generating Summary
"""

# %%
print keywords(sentences)