# import libraries
from sklearn.feature_extraction.text import CountVectorizer

# örnek metin
docs=[
    "Bu çalışma NGram çalışmasıdır.",
    "Bu çalışma doğal dil işleme çalışmasıdır."
]

# unigram, bigram ve trigram şeklinde 3 farklı N değerine sahip gram modeli
vectorizer_unigram = CountVectorizer(ngram_range=(1,1))
vectorizer_bigram = CountVectorizer(ngram_range=(2,2))
vectorizer_trigram = CountVectorizer(ngram_range=(3,3))

#unigram
x_unigram = vectorizer_unigram.fit_transform(docs)
unigram_features = vectorizer_unigram.get_feature_names_out()

#bigram
x_bigram = vectorizer_bigram.fit_transform(docs)
bigram_features = vectorizer_bigram.get_feature_names_out()

#trigram
x_trigram = vectorizer_trigram.fit_transform(docs)
trigram_features = vectorizer_trigram.get_feature_names_out()

# sonuçların incelemesi
print(f"Unigram: {unigram_features}")
print(f"Bigram: {bigram_features}")
print(f"Trigram: {trigram_features}")