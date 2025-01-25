# kütüphanelerin içe aktarılması
import pandas as pd
from data_cleaning import clean_text
from sklearn.feature_extraction.text import TfidfVectorizer


# veri setinin içe aktarılması
df=pd.read_csv("IMDB Dataset.csv")

# metinlerin temizlenmesi
cleaned_docs = [clean_text(doc) for doc in df.review]

# tf-idf in tanımlanması
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(cleaned_docs)

# kelime kümesinin oluşturulması
features_names = vectorizer.get_feature_names_out()
tfidf_score = X.mean(axis =0).A1

# tf-idf skorlarının dataframe e dönüştürülmesi
df_tfidf = pd.DataFrame({"word":features_names,"tfidf":tfidf_score})

# skorların sıralanması ve ekrana yazdırılması
df_tfidf_sorted = df_tfidf.sort_values(by="tfidf",ascending=False)
print(df_tfidf_sorted.head(10))