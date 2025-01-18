import nltk
nltk.download("wordnet") #kelime köklerini bulmak için gerekli olan veritabanı


from nltk.stem import PorterStemmer #stemming işlemi için gerekli fonksiyon
stemmer = PorterStemmer()
words = ["running","runner","ran","runs","better","go","went"]
#kelime köklerini bulmak için kullanılır, bunu yaparken stemmerin stem fonksiyonunu kullanırız
stems=[stemmer.stem(word) for word in words] 
print(f"stems: {stems}")



from nltk.stem import WordNetLemmatizer #lemmatization işlemi için gerekli fonksiyon
lemmatizer = WordNetLemmatizer()
words = ["running","runner","ran","runs","better","go","went"]
lemmas= [lemmatizer.lemmatize(word,pos="v") for word in words]
print(f"lemmas: {lemmas}")



def stemming_lemmatization(text):
    words = nltk.word_tokenize(text)
    stems=[stemmer.stem(word) for word in words]
    lemmas= [lemmatizer.lemmatize(word,pos="v") for word in words]
    return stems, lemmas

