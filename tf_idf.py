#import libraries
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer

# örnek belge oluştur
docs=[
    "Köpek çok tatlı bir hayvandır",
    "Köpek ve kuşlar çok tatlı hayvanlardır.",
    "Inekler süt üretirler"
]


# vektorizer tanımla
tfidf_vectorizer = TfidfVectorizer()


# metinleri sayısal hale getir
X=tfidf_vectorizer.fit_transform(docs)

# kelime kümesini incele
features_names=tfidf_vectorizer.get_feature_names_out()

# vektor temsilini incele
vektor_temsili = X.toarray()
print(f"tf-idf: {vektor_temsili}")

df_tfidf = pd.DataFrame(vektor_temsili, columns=features_names)

# ortalama tf idf değerini incele

tf_idf = df_tfidf.mean(axis=0)