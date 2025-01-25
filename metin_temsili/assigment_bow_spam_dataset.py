# kütüphanelerin içe aktarılması
import pandas as pd 
from sklearn.feature_extraction.text import CountVectorizer
from data_cleaning import clean_text
from collections import Counter

# veri setinin içe aktarılamsı
df = pd.read_csv("spam.csv", encoding='latin-1')

# metinlerin alınması
docs = df["v2"]

# metinlerin temizlenmesi
cleaned_docs=[clean_text(doc) for doc in docs]

# vectorizerin tanımlanması
vectorizer = CountVectorizer()

# metnin sayısal hale getirilmesi
X=vectorizer.fit_transform(cleaned_docs)

# kelime kümesinin alınması
fearures_names = vectorizer.get_feature_names_out()

# kelime frekanslarının elde edilmesi
word_counts=X.sum(axis=0).A1
word_freq = dict(zip(fearures_names,word_counts))

# ilk 10 kelimenin ekrana yazdırılması
most_common_first_10_words = Counter(word_freq).most_common(10)
print(f"Most common first 10 words: {most_common_first_10_words}")
