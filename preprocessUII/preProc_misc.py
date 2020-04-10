import pandas as pd
from nltk.corpus import stopwords
from nltk import word_tokenize
import os.path

my_path = os.path.abspath(os.path.dirname(__file__))
path = os.path.join(my_path, "./data/colloquial-indonesian-lexicon.csv")

stop_words = set(stopwords.words('indonesian'))

df_crc = pd.read_csv(path)[['slang','formal','category1']]
df_crc = df_crc[df_crc['category1']=='elongasi']

def koreksi_elongasi(word, df_crc=df_crc):
    if list(df_crc['formal'][df_crc['slang']=='{}'.format(word)].values) == []:
        return word
    return df_crc['formal'][df_crc['slang']=='{}'.format(word)].values[0]


def removeStopword(text, stop_words=stop_words): # membuang kata2 yang terdapat pada stopwords id
    word_tokens = word_tokenize(text)
    filtered_sentence = [w for w in word_tokens if not w in stop_words]
    return ' '.join(filtered_sentence)