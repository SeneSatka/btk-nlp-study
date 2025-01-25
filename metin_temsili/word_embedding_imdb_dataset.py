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


# word2vec model tanımlama
model=Word2Vec(sentences=tokenized_docs,vector_size=50,window=5,min_count=1,sg=0)
word_vectors=model.wv

words = list(word_vectors.index_to_key)
vectors = [word_vectors[word] for word in words]

#clustering KMeans K=2
kmeans=KMeans(n_clusters=4)
kmeans.fit(vectors)
clusters=kmeans.labels_

# PCA 50 -> 2
pca=PCA(n_components=2)
reduced_vectors=pca.fit_transform(vectors)


# 2 boyutlu görselleştirme
plt.figure()
plt.scatter(reduced_vectors[:,0],reduced_vectors[:,1],c=clusters,cmap="viridis")

centers= pca.transform(kmeans.cluster_centers_)
plt.scatter(centers[:,0],centers[:,1],c="red",marker="x",s=170,label="Center")
plt.legend()

for i,word in enumerate(words):
    plt.text(reduced_vectors[i,0],reduced_vectors[i,1],word,fontsize=7)



plt.show()
