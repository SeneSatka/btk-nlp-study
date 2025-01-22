from bs4 import BeautifulSoup
from textblob import TextBlob
import re
import string
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def clean_text(text):
    nt=text.lower()
    soup = BeautifulSoup(nt, 'html.parser')
    nt = soup.get_text()
    nt =re.sub(r"\d+", "", nt)
    nt = re.sub(r'[^a-zA-Z0-9\s]', '', nt)
    nt = nt.translate(str.maketrans('', '', string.punctuation))
    nt = TextBlob(nt).correct()
    nt = " ".join([word for word in nt.split() if len(word)>2 and word not in stop_words])
    return nt
    