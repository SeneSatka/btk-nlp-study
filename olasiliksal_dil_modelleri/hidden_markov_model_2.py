# kütüphaneleri içe aktar
import nltk
from nltk.tag import hmm
from nltk.corpus import conll2000
# veri setini içe aktarma
nltk.download("conll2000")
train_data = conll2000.tagged_sents("train.txt")
test_data = conll2000.tagged_sents("test.txt")
# train HMM
trainer = hmm.HiddenMarkovModelTrainer()
hmm_tagger = trainer.train(train_data)

# yeni cümle ve test

test_sentence= "I like going to school".split()
tags = hmm_tagger.tag(test_sentence)

print(tags)