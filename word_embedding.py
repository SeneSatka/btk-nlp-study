#import libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA # Principal Component Analysis: boyut azaltma

from gensim.models import Word2Vec,FastText
from gensim.utils import simple_preprocess

# örnek veri seti oluşturma
sentences=[
    "Köpek çok tatlı bir hayvandır.",
    "Köpekler evcil hayvanlardır.",
    "Kediler genellikle yaramaz davranışlarda bulunurlar.",
    "Köpekler sadık ve dost canlısı hayvanlardır.",
    "Hayvanlar insanlar için iyi arkadaşlardır."]

tokenized_sentences=[simple_preprocess(sentence) for sentence in sentences]
# word2vec
word2_vec_model=Word2Vec(sentences=tokenized_sentences,vector_size=50,window=5,min_count=1,sg=0)


# fasttext
fast_text_model= FastText(sentences=tokenized_sentences,vector_size=50,window=5,min_count=1,sg=0)


# görselleştirme: PCA
def plot_word_embeding(model,title):
    word_vectors=model.wv
    words = list (word_vectors.index_to_key)[:1000]
    vectors=[word_vectors[word] for word in words]
    
    #PCA
    pca = PCA(n_components=3)
    reduced_vectors=pca.fit_transform(vectors)
    
    
    # 3d görselleştiröe
    fig=plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111,projection="3d")
    
    #vektörleri çizme
    ax.scatter(reduced_vectors[:,0],reduced_vectors[:,1],reduced_vectors[:,2])
    
    #kelimeleri etiketleme
    for i, word in enumerate(words):
        ax.text(reduced_vectors[i,0],reduced_vectors[i,1],reduced_vectors[1,2],word,fontsize=12)
        
    ax.set_title(title)
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.set_zlabel("Component 3")
    plt.show()
    
plot_word_embeding(word2_vec_model,"Word2Vec")
plot_word_embeding(fast_text_model,"FastText")