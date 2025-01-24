#import libraries
import pandas as pd
import matplotlib.pyplot as plt
from data_cleaning import clean_text
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess


#veri seti yükleme
df=pd.read_csv("IMDB Dataset.csv")
docs = df["review"]

#metin temizleme
cleaned_docs=[clean_text(sentence) for sentence in docs]

#metin tokenization
tokenized_docs=[simple_preprocess(doc) for doc in cleaned_docs]


