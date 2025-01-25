#import libraries
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from metin_on_isleme.data_cleaning import clean_text
#veri seti yükle
df = pd.read_csv("spam.csv", encoding='latin-1')
cleaned_texts = [clean_text(text) for text in df.v2]
#tfidf
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(cleaned_texts)


#kelime kümesini incele
features_names=vectorizer.get_feature_names_out()
tfidf_score = X.mean(axis=0).A1 #her kelimenin tfidf skorunu hesapla

#tfidf skorlarını içeren bir df oluştur
df_tfidf = pd.DataFrame({"word":features_names, "tfidf":tfidf_score})

#skorları sırala ve sonuçları incele
df_tfidf_sorted = df_tfidf.sort_values(by="tfidf", ascending=False)
print(df_tfidf_sorted.head(10))