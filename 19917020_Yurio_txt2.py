import pandas as pd
from nltk import ngrams
from sklearn.feature_extraction.text import CountVectorizer
import warnings 
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

from preprocessUII.spellchecker import correction
from preprocessUII.preProc_misc import stop_words, removeStopword, koreksi_elongasi

stop_words.add('of')
stop_words.add('the')
stop_words.add('jpg')
stop_words.add('jmpl')
stop_words.add('px')
stop_words.add('cd')

def get_top_n_words(corpus, ngr=1, n=None):   
    """get_top_n_words for any n-gram you decide
    
    :param corpus: corpora of text you've created before , usually formatted as list of sentence
    :type corpus: list
    :param ngr: deciding n-gram you tend to build, defaults to 1
    :type ngr: int, optional
    :param n: deciding top n words like top 10 or top 20 of word or gram, defaults to None
    :type n: int, optional
    :return: list of tuple from word and frequency
    :rtype: list of tuple
    """
    vec = CountVectorizer(ngram_range=(ngr, ngr)).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]

# 2. Koreksi ejaan untuk menangani salah ketik (Misal: yagn --> dikoreksi menjadi yang)
# 3. Word normalization untuk menangani lenghtening word (Misal: horeeee --> dinormalisasi menjadi hore)

print('koreksi : ', correction('yagn')) # percobaan pertama menggunakan correction function 
print('koreksi : ', correction('horeeee'))

print('koreksi : ', koreksi_elongasi('horeeee')) # percobaan kedus menggunakan koreksi_elongasi function

# Buatlah code dengan Python untuk menangani permasalahan berikut:
# 1. Generate n-gram pada suatu corpus

sentence = "akankah diri ini terus bersamamu disaat orang lain sudah menjadi tuanmu"

n = 6
sixgrams = ngrams(sentence.split(), n)

for grams in sixgrams: # contoh simple proses n-gramisasi
    print(grams)

corpus_wiki = open('idwiki_1k.txt').read().split("\n") # corpus indonesia wikipedia 1000 article
corpus_wiki_sw = [removeStopword(article) for article in corpus_wiki] # corpus indonesia wikipedia 1000 article yang sudah 
                                                                      # dibersihkan dari stopwords


############### unigram non stop words #################################
common_words = get_top_n_words(corpus_wiki, ngr=1, n=20) # menggunakan get_top_n_words untuk menganalisa dominasi frase 
                                                         # n-gram pada suatu corpus

for word, freq in common_words: # melihat top 20 word
    print(word, freq)

df2 = pd.DataFrame(common_words, columns = ['ArticleText' , 'count'])

matplotlib.rcParams['figure.figsize'] = (14, 10) # melihat top 20 word visualisasi dengan matplotlib
ax = df2.plot.bar(x='ArticleText', y='count', rot='vertical')
plt.show()

ax = sns.barplot(x='ArticleText', y='count', data=df2) # melihat top 20 word visualisasi dengan seaborn 
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
plt.show()

############### unigram stop words ################################# Pengulangan pd corpus stopwords and so on....
common_words = get_top_n_words(corpus_wiki_sw, ngr=1, n=20)

for word, freq in common_words:
    print(word, freq)

df2 = pd.DataFrame(common_words, columns = ['ArticleText' , 'count'])

ax = df2.plot.bar(x='ArticleText', y='count', rot='vertical')
plt.show()

ax = sns.barplot(x='ArticleText', y='count', data=df2)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
plt.show()

############### bigram non stop words #################################
common_words = get_top_n_words(corpus_wiki, ngr=2, n=20)

for word, freq in common_words:
    print(word, freq)

df2 = pd.DataFrame(common_words, columns = ['ArticleText' , 'count'])

ax = df2.plot.bar(x='ArticleText', y='count', rot='vertical')
plt.show()

ax = sns.barplot(x='ArticleText', y='count', data=df2)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
plt.show()

############### bigram stop words #################################
common_words = get_top_n_words(corpus_wiki_sw, ngr=2, n=20)

for word, freq in common_words:
    print(word, freq)

df2 = pd.DataFrame(common_words, columns = ['ArticleText' , 'count'])

ax = df2.plot.bar(x='ArticleText', y='count', rot='vertical')
plt.show()

ax = sns.barplot(x='ArticleText', y='count', data=df2)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
plt.show()

############### trigram non stop words #################################
common_words = get_top_n_words(corpus_wiki, ngr=3, n=20)

for word, freq in common_words:
    print(word, freq)

df2 = pd.DataFrame(common_words, columns = ['ArticleText' , 'count'])

ax = df2.plot.bar(x='ArticleText', y='count', rot='vertical')
plt.show()

ax = sns.barplot(x='ArticleText', y='count', data=df2)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
plt.show()

############### trigram stop words #################################
common_words = get_top_n_words(corpus_wiki_sw, ngr=3, n=20)

for word, freq in common_words:
    print(word, freq)

df2 = pd.DataFrame(common_words, columns = ['ArticleText' , 'count'])

ax = df2.plot.bar(x='ArticleText', y='count', rot='vertical')
plt.show()

ax = sns.barplot(x='ArticleText', y='count', data=df2)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
plt.show()

############### 4-gram non stop words #################################
common_words = get_top_n_words(corpus_wiki, ngr=4, n=20)

for word, freq in common_words:
    print(word, freq)

df2 = pd.DataFrame(common_words, columns = ['ArticleText' , 'count'])

ax = df2.plot.bar(x='ArticleText', y='count', rot='vertical')
plt.show()

ax = sns.barplot(x='ArticleText', y='count', data=df2)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
plt.show()

############### 4-gram stop words #################################
common_words = get_top_n_words(corpus_wiki_sw, ngr=4, n=20)

for word, freq in common_words:
    print(word, freq)

df2 = pd.DataFrame(common_words, columns = ['ArticleText' , 'count'])

ax = df2.plot.bar(x='ArticleText', y='count', rot='vertical')
plt.show()

ax = sns.barplot(x='ArticleText', y='count', data=df2)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
plt.show()



