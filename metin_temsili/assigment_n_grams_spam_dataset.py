# kütüphanelerin içe aktarılması
import pandas as pd
from data_cleaning import clean_text
from sklearn.feature_extraction.text import CountVectorizer

# veri setinin içe aktarılması
docs = pd.read_csv( "spam.csv",encoding='latin-1')

# metinlerin alınması 
texts = docs.v2

# metinlerin temizlenmesi
cleaned_texts=[clean_text(text) for text in texts]

# unigram vectorizerin tanımlanması ve kümesinin oluşturulması
vectorizer_unigram = CountVectorizer(ngram_range=(1,1))
X_unigram = vectorizer_unigram.fit_transform(cleaned_texts)
features_unigram=vectorizer_unigram.get_feature_names_out()

# bigram vectorizerin tanımlanması ve kümesinin oluşturulması
vectorizer_bigram = CountVectorizer(ngram_range=(2,2))
X_bigram = vectorizer_bigram.fit_transform(cleaned_texts)
features_bigram=vectorizer_bigram.get_feature_names_out()

#trigram vectorizerin tanımlanması ve kümesinin oluşturulması
vectorizer_trigram = CountVectorizer(ngram_range=(3,3))
X_trigram = vectorizer_trigram.fit_transform(cleaned_texts)
features_trigram=vectorizer_trigram.get_feature_names_out()
# sonuçların ekrana yazdırılması

print(f"Unigram {features_unigram}")
print(f"Bigram {features_bigram}")
print(f"Trigram {features_trigram}")