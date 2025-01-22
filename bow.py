#count vectorizer içeriye aktar
from sklearn.feature_extraction.text import CountVectorizer

#veri seti oluştur
doc=[
    "kedi bahçede",
    "kedi evde"
]
# vectorizer tanımla

vectorizer = CountVectorizer()

# metni sayısal vektörlere çevir
X = vectorizer.fit_transform(doc)

# kelime kümesi oluşturma [bashçede, evde, kedi]
feature_names = vectorizer.get_feature_names_out() # kelime kümesini oluşturma
print(f"kelime kümesi: {feature_names}")

#vektor temsili
vector_temsili=X.toarray()   

print(f"vektor temsili: {vector_temsili}")