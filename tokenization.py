import nltk #natural language toolkit
nltk.download('punkt_tab') # metni kelime ve cümlele bazında tokenlere ayırmak için gerekli
nltk.download('punkt') # metni kelime ve cümlele bazında tokenlere ayırmak için gerekli
text="Hello, World! How are you? Hello, hi ..."

#kelime bazında tokenlere ayırmak için kullanılır
word_tokens = nltk.word_tokenize(text)

#cümle bazında tokenlere ayırmak için kullanılır
sentence_tokens = nltk.sent_tokenize(text)


