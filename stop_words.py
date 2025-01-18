import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')# farklı dillerdeki stop words listelerini indirmek için kullanılır


# ingilizce stop words analizi

stop_words_eng = set(stopwords.words('english'))

#ornek ingilizce metin
text = "There are some examples of handling stop words from some texts."
text_list = text.split()
#eğer kelime stop words listesinde yoksa onu filtered_words listesine ekler
filtered_words_eng=[word for word in text_list if word.lower() not in stop_words_eng]
print(f"Eng_filtered_words: {filtered_words_eng}")


#türkçe stop words analizi
stop_words_tr = set(stopwords.words('turkish'))

#ornek türkçe metin
text = "Merhaba arkadaşlar çok güzel bir ders işliyoruz."
metin_list = text.split()
#eğer kelime stop words listesinde yoksa onu filtered_words listesine ekler
filtered_words_tr=[word for word in metin_list if word.lower() not in stop_words_tr]
print(f"Tr_filtered_words: {filtered_words_tr}")

#kutuphanesiz stop words analizi

#stop words listesi
tr_stop_words=["için","bu","ile","mu","mi","özel"]

metin="Bu bir denemedir. Amacımız bu metinde bulunan özel karakterleri elemek mi acaba?"

filtered_words=[word for word in metin.split() if word.lower() not in tr_stop_words]
filtered_stop_words=set([word.lower() for word in metin.split() if word.lower() in tr_stop_words])
print(f"Filtered_words: {filtered_words}")
print(f"Filtered_stop_words: {filtered_stop_words}")