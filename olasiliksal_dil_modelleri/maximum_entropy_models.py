"""
classification problemi: duygu analizi -> olumlu veya olumsuz sınıflandrma
"""

# kütüphanelerin içe aktarılması
from nltk.classify import MaxentClassifier

# veri setinin oluşturulması
train_data =[
    ({"love":True,"amazing":True,"happy":True,"terrible":False},"positive"),
    ({"hate":True,"terrible":True},"negative"),
    ({"joy":True,"happy":True,"hate":False},"positive"),
    ({"sad":True,"depressed":True,"love":False},"negative"),
]

# train 
classifier = MaxentClassifier.train(train_data,max_iter=10)

# yeni cümle ile test
test_sentence = "I love this movie and it was terrible"
features = {word:(word in test_sentence.lower().split()) for word in ["love","amazing","terrible","happy","joy","hate","sad","depressed","love"]}
label =classifier.classify(features)
print(label)