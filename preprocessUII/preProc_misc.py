import pandas as pd
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk import word_tokenize
import os.path

# initiate dictionary for normalize text
my_path = os.path.abspath(os.path.dirname(__file__))
path = os.path.join(my_path, "./data/colloquial-indonesian-lexicon.csv")

# initiate stopwords from NLTK
stop_words = set(stopwords.words('indonesian'))

# initiate dataframe for mapping n normalize text
df_crc = pd.read_csv(path)[['slang','formal','category1']]
df_crc = df_crc[df_crc['category1']=='elongasi']

def koreksi_elongasi(word, df_crc=df_crc):
    """koreksi elongasi sebagai tahapan normalisasi text
    
    :param word: kata yang akan di normalize
    :type word: string
    :param df_crc: dataframe correction for normalize, defaults to df_crc
    :type df_crc: dataframe pandas, optional
    :return: normalized text after mapping 
    :rtype: string
    """
    if list(df_crc['formal'][df_crc['slang']=='{}'.format(word)].values) == []:
        return word
    return df_crc['formal'][df_crc['slang']=='{}'.format(word)].values[0]


def removeStopword(text, stop_words=stop_words): 
    """membuang kata2 yang terdapat pada stopwords id
    
    :param text: list kata yang akan dibuang dari daftar stopwords yang ada
    :type text: list of string
    :param stop_words: set of stopwords from NLTK indonesian, defaults to stop_words initiate in beginning
    :type stop_words: dataframe pandas, optional
    :return: list of string after removing stop words
    :rtype: list
    """
    word_tokens = word_tokenize(text)
    filtered_sentence = [w for w in word_tokens if not w in stop_words]
    return ' '.join(filtered_sentence)