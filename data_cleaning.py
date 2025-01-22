from bs4 import BeautifulSoup
from textblob import TextBlob
import re
import string

def data_cleaning(text):
    nt=text.lower()
    soup = BeautifulSoup(text, 'html.parser')
    nt = soup.get_text()
    nt = re.sub(r'[^a-zA-Z0-9\s]', '', nt)
    nt = nt.translate(str.maketrans('', '', string.punctuation))
    nt = TextBlob(nt).correct()
    return nt
    