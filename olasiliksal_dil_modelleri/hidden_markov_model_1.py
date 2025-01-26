"""
Part Of Speech POS: kelimelerin uygun sözcük türünü bulma çalışması
HMM
I(Zamir) am ateacher (isim)
"""
# kütüphaneleri içe aktar
import nltk
from nltk.tag import hmm

# örnek training data tanımlama
train_data =[
    [("I","PRP"),("am","VBP"),("a","DT"),("teacher","NN")],
    [("You","PRP"),("are","VBP"),("a","DT"),("student","NN")]
]

# train HMM
trainer = hmm.HiddenMarkovModelTrainer()
hmm_tagger = trainer.train(train_data)

# yeni cümle ile cümlede bulunan her sözcüğün tüünü etiketleme

test_sentence= "I am a student".split()
tags = hmm_tagger.tag(test_sentence)

print(tags)