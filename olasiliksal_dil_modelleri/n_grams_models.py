# kütüphanaleri içe aktarma
import nltk
from nltk.util import ngrams
from nltk.tokenize import word_tokenize

from collections import Counter

# örnek veri seti oluştur
corpus =[
    "I love apple",
    "I love him",
    "I love NLP",
    "You love me",
    "He loves apple",
    "They love apple",
    "I love you and you love me",
]
"""
    problem tanımı:
    dil modeli yapmak istiyoruz
    amaç 1 kelimeden sonra gelecek kelimeyi tahmin etmek: metin oluşturma
    bunun için n gram dil modeli kullanacağız
"""

# verileri token haline getir
tokens=[word_tokenize(sentence.lower()) for sentence in corpus]


# bigram
bigrams =[]
for token_list in tokens:
    bigrams.extend(list(ngrams(token_list,2)))

bigrams_freq=Counter(bigrams)
print(bigrams_freq)

# trigram
trigram =[]
for token_list in tokens:
    trigram.extend(list(ngrams(token_list,3)))

trigram_freq=Counter(trigram)
print(trigram_freq)

# modeli test etme

# "I love" bigramından sonra "you" veya "apple" kelimelerinin gelme olasılıklarını hesaplama

bigram =("i","love") # hedef bigram
# i love you olma olasılığı
prob_you = trigram_freq[("i","love","you")]/bigrams_freq[bigram]
print(f"i love dan sonra you gelme ihtmiali {prob_you}")


# i love apple olma olasılığı
prob_apple = trigram_freq[("i","love","apple")]/bigrams_freq[bigram]
print(f"i love dan sonra apple gelme ihtmiali {prob_apple}")
