#import libraries
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from metin_on_isleme.data_cleaning import clean_text
from collections import Counter
import re

# veri setinin içeriye aktarılması
df = pd.read_csv("IMDB Dataset.csv")

# metin verilerinin alınması
docs = df['review']
labels = df['sentiment'] # positive or negative

# metin temizleme


#metinleri temizleme
cleaned_doc = [clean_text(doc) for doc in docs[:100]]

###  bow

#vectorizer tanımla

vectorizer = CountVectorizer()

# metin -> sayısal hale getir
X = vectorizer.fit_transform(cleaned_doc)

# kelime kümesi göster
features_names = vectorizer.get_feature_names_out()


# vektor temsili göster
vektor_temsili = X.toarray()

df_bow=pd.DataFrame(vektor_temsili, columns=features_names)

# kelime frekasnlarını göster

word_counts=X.sum(axis=0).A1
word_freq = dict(zip(features_names, word_counts))

# ilk 5 kelimeyi print ettir
most_common_words = Counter(word_freq).most_common(5)
print(f"Most common words: {most_common_words}")