

Upgrade
YurioWindiatmoko
Towards Data Science
DATA SCIENCE
MACHINE LEARNING
PROGRAMMING
VISUALIZATION
AI
VIDEO
ABOUT
CONTRIBUTE

Photo credit: Pixabay
A Complete Exploratory Data Analysis and Visualization for Text Data
How to combine visualization and NLP in order to generate insights in an intuitive way
Susan Li
Susan Li
Follow
Mar 19, 2019 · 8 min read



Visually representing the content of a text document is one of the most important tasks in the field of text mining. 
As a data scientist or NLP specialist, not only we 
- explore the content of documents from different aspects and at different levels of details, but also we 
- summarize a single document, 
- show the words and topics, 
- detect events, and 
- create storylines.

However, there are some gaps between visualizing unstructured (text) data and structured data. 
For example, 

many text visualizations do not represent the text directly, they `represent an output of a language model`(
- `word count`, 
- `character length`, 
- word sequences, etc.).

In this post, we will use Womens Clothing E-Commerce Reviews data set, and try to explore and visualize as much as we can, using `Plotly’s Python graphing library` and `Bokeh visualization library`. 

Not only we are going to explore text data, but also we will 
- `visualize numeric` and 
- `categorical features`. 

Let’s get started!
The Data

    df = pd.read_csv('Womens Clothing E-Commerce Reviews.csv')

<p align="center">
  <img src="https://miro.medium.com/max/1104/1*1E-zIJXMas05676qvuWSzw.png">
</p>

table 1

After a brief inspection of the data, we found there are a series of data pre-processing we have to conduct.
- Remove the “Title” feature.
- Remove the rows where “Review Text” were missing.
- Clean “Review Text” column.

`Using TextBlob to calculate sentiment polarity which lies in the range of [-1,1]` 
- where 1 means positive sentiment and 
- -1 means a negative sentiment.
- Create new feature for the length of the review.
- Create new feature for the word count of the review.

```python
df.drop('Unnamed: 0', axis=1, inplace=True)
df.drop('Title', axis=1, inplace=True)
df = df[~df['Review Text'].isnull()]

def preprocess(ReviewText):
    ReviewText = ReviewText.str.replace("(<br/>)", "")
    ReviewText = ReviewText.str.replace('(<a).*(>).*(</a>)', '')
    ReviewText = ReviewText.str.replace('(&amp)', '')
    ReviewText = ReviewText.str.replace('(&gt)', '')
    ReviewText = ReviewText.str.replace('(&lt)', '')
    ReviewText = ReviewText.str.replace('(\xa0)', ' ')  
    return ReviewText
df['Review Text'] = preprocess(df['Review Text'])

df['polarity'] = df['Review Text'].map(lambda text: TextBlob(text).sentiment.polarity)
df['review_len'] = df['Review Text'].astype(str).apply(len)
df['word_count'] = df['Review Text'].apply(lambda x: len(str(x).split()))
```
text_preprocessing.py

To preview whether the sentiment polarity score works, we randomly select 5 reviews with the highest sentiment polarity score (1):

    print('5 random reviews with the highest positive sentiment polarity: \n')
    cl = df.loc[df.polarity == 1, ['Review Text']].sample(5).values
    for c in cl:
        print(c[0])

Figure 1

Then randomly select 5 reviews with the most neutral sentiment polarity score (zero):

    print('5 random reviews with the most neutral sentiment(zero) polarity: \n')
    cl = df.loc[df.polarity == 0, ['Review Text']].sample(5).values
    for c in cl:
        print(c[0])

Figure 2

There were only 2 reviews with the most negative sentiment polarity score:

    print('2 reviews with the most negative polarity: \n')
    cl = df.loc[df.polarity == -0.97500000000000009, ['Review Text']].sample(2).values
    for c in cl:
        print(c[0])

Figure 3

It worked!

Univariate visualization with Plotly
Single-variable or univariate visualization is the simplest type of visualization which consists of observations on only a single characteristic or attribute. 
Univariate visualization includes 
- histogram, 
- bar plots and 
- line charts.
The distribution of review sentiment polarity score

        df['polarity'].iplot(
            kind='hist',
            bins=50,
            xTitle='polarity',
            linecolor='black',
            yTitle='count',
            title='Sentiment Polarity Distribution')

Figure 4

Vast majority of the sentiment polarity scores are greater than zero, means most of them are pretty positive.

The distribution of review ratings

    df['Rating'].iplot(
        kind='hist',
        xTitle='rating',
        linecolor='black',
        yTitle='count',
        title='Review Rating Distribution')

Figure 5

The ratings are in align with the polarity score, that is, most of the ratings are pretty high at 4 or 5 ranges.

The distribution of reviewers age

    df['Age'].iplot(
        kind='hist',
        bins=50,
        xTitle='age',
        linecolor='black',
        yTitle='count',
        title='Reviewers Age Distribution')

Figure 6

Most reviewers are in their 30s to 40s.


The distribution review text lengths

    df['review_len'].iplot(
        kind='hist',
        bins=100,
        xTitle='review length',
        linecolor='black',
        yTitle='count',
        title='Review Text Length Distribution')

Figure 7

The distribution of review word count

    df['word_count'].iplot(
        kind='hist',
        bins=100,
        xTitle='word count',
        linecolor='black',
        yTitle='count',
        title='Review Text Word Count Distribution')

Figure 8

There were quite number of people like to leave long reviews.

For categorical features, we simply use bar chart to present the frequency.


The distribution of division

    df.groupby('Division Name').count()['Clothing ID'].iplot(kind='bar', yTitle='Count', linecolor='black', opacity=0.8,
    title='Bar chart of Division Name', xTitle='Division Name')

Figure 9

General division has the most number of reviews, and 
Initmates division has the least number of reviews.

The distribution of department

    df.groupby('Department Name').count()['Clothing ID'].sort_values(ascending=False).iplot(kind='bar', yTitle='Count', linecolor='black', opacity=0.8,
    title='Bar chart of Department Name', xTitle='Department Name')

Figure 10

When comes to department, Tops department has the most reviews and Trend department has the least number of reviews.

The distribution of class

    df.groupby('Class Name').count()['Clothing ID'].sort_values(ascending=False).iplot(kind='bar', yTitle='Count', linecolor='black', opacity=0.8,
    title='Bar chart of Class Name', xTitle='Class Name')

Figure 11


Now we come to “Review Text” feature, before explore this feature, we need to 
- `extract N-Gram features`. 
`N-grams are used` `to describe` `the number of words` `used as observation points`, 
e.g., 
- `unigram` `means singly-worded`, 
- `bigram means 2-worded` `phrase`, and 
- `trigram means 3-worded` `phrase`. 

In order to do this, we use `scikit-learn’s CountVectorizer` function.

First, it would be interesting to 
`compare unigrams` `before` and `after` `removing stop words`.

The distribution of top unigrams before removing stop words

```python

def get_top_n_words(corpus, n=None):
    vec = CountVectorizer().fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]

common_words = get_top_n_words(df['Review Text'], 20)

for word, freq in common_words:
    print(word, freq)

df1 = pd.DataFrame(common_words, columns = ['ReviewText' , 'count'])

df1.groupby('ReviewText').sum()['count'].sort_values(ascending=False).iplot(
    kind='bar', yTitle='Count', linecolor='black', title='Top 20 words in review before removing stop words')
```

top_unigram.py

Figure 12
The distribution of top unigrams after removing stop words

```python
from sklearn.feature_extraction.text import CountVectorizer
import warnings 
warnings.filterwarnings('ignore')
from plotly.offline import iplot

def get_top_n_words(corpus, n=None):
    vec = CountVectorizer(stop_words = 'english').fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]

common_words = get_top_n_words(df['Review Text'], 20)

for word, freq in common_words:
    print(word, freq)

df2 = pd.DataFrame(common_words, columns = ['ReviewText' , 'count'])

df2.groupby('ReviewText').sum()['count'].sort_values(ascending=False).iplot(
    kind='bar', yTitle='Count', linecolor='black', title='Top 20 words in review after removing stop words')
```
top_unigram_no_stopwords.py


Figure 13

Second, we want to compare bigrams before and after removing stop words.

The distribution of top bigrams before removing stop words

top_bigram.py

Figure 14
The distribution of top bigrams after removing stop words

top_bigram_no_stopwords.py

Figure 15
Last, we compare trigrams before and after removing stop words.
The distribution of Top trigrams before removing stop words

top_trigram.py

Figure 16
The distribution of Top trigrams after removing stop words

top_trigram_no_stopwords.py

Figure 17
Part-Of-Speech Tagging (POS) is a process of assigning parts of speech to each word, such as noun, verb, adjective, etc
We use a simple TextBlob API to dive into POS of our “Review Text” feature in our data set, and visualize these tags.
The distribution of top part-of-speech tags of review corpus

POS.py

Figure 18
Box plot is used to compare the sentiment polarity score, rating, review text lengths of each department or division of the e-commerce store.
What do the departments tell about Sentiment polarity

department_polarity.py

Figure 19
The highest sentiment polarity score was achieved by all of the six departments except Trend department, and the lowest sentiment polarity score was collected by Tops department. And the Trend department has the lowest median polarity score. If you remember, the Trend department has the least number of reviews. This explains why it does not have as wide variety of score distribution as the other departments.
What do the departments tell about rating

rating_division.py

Figure 20
Except Trend department, all the other departments’ median rating were 5. Overall, the ratings are high and sentiment are positive in this review data set.
Review length by department

length_department.py

Figure 21
The median review length of Tops & Intimate departments are relative lower than those of the other departments.
Bivariate visualization with Plotly
Bivariate visualization is a type of visualization that consists two features at a time. It describes association or relationship between two features.
Distribution of sentiment polarity score by recommendations

polarity_recommendation.py

Figure 22
It is obvious that reviews have higher polarity score are more likely to be recommended.
Distribution of ratings by recommendations

rating_recommendation.py

Figure 23
Recommended reviews have higher ratings than those of not recommended ones.
Distribution of review lengths by recommendations

review_length_recommend.py

Figure 24
Recommended reviews tend to be lengthier than those of not recommended reviews.
2D Density jointplot of sentiment polarity vs. rating

sentiment_polarity_rating.py

Figure 24
2D Density jointplot of age and sentiment polarity

age_polarity.py

Figure 25
There were few people are very positive or very negative. People who give neutral to positive reviews are more likely to be in their 30s. Probably people at these age are likely to be more active.
Finding characteristic terms and their associations
Sometimes we want to analyzes words used by different categories and outputs some notable term associations. We will use scattertext and spaCy libraries to accomplish these.
First, we need to turn the data frame into a Scattertext Corpus. To look for differences in department name, set the category_colparameter to 'Department Names', and use the review present in the Review Text column, to analyze by setting the text col parameter. Finally, pass a spaCy model in to the nlp argument and call build() to construct the corpus.
Following are the terms that differentiate the review text from a general English corpus.
corpus = st.CorpusFromPandas(df, category_col='Department Name', text_col='Review Text', nlp=nlp).build()
print(list(corpus.get_scaled_f_scores_vs_background().index[:10]))

Figure 26
Following are the terms in review text that are most associated with the Tops department:
term_freq_df = corpus.get_term_freq_df()
term_freq_df['Tops Score'] = corpus.get_scaled_f_scores('Tops')
pprint(list(term_freq_df.sort_values(by='Tops Score', ascending=False).index[:10]))

Figure 27
Following are the terms that are most associated with the Dresses department:
term_freq_df['Dresses Score'] = corpus.get_scaled_f_scores('Dresses')
pprint(list(term_freq_df.sort_values(by='Dresses Score', ascending=False).index[:10]))

Figure 28
Topic Modeling Review Text
Finally, we want to explore topic modeling algorithm to this data set, to see whether it would provide any benefit, and fit with what we are doing for our review text feature.
We will experiment with Latent Semantic Analysis (LSA) technique in topic modeling.
Generating our document-term matrix from review text to a matrix of TF-IDF features.
LSA model replaces raw counts in the document-term matrix with a TF-IDF score.
Perform dimensionality reduction on the document-term matrix using truncated SVD.
Because the number of department is 6, we set n_topics=6.
Taking the argmax of each review text in this topic matrix will give the predicted topics of each review text in the data. We can then sort these into counts of each topic.
To better understand each topic, we will find the most frequent three words in each topic.

topic_model_LSA.py

Figure 29
top_3_words = get_top_n_words(3, lsa_keys, document_term_matrix, tfidf_vectorizer)
labels = ['Topic {}: \n'.format(i) + top_3_words[i] for i in lsa_categories]
fig, ax = plt.subplots(figsize=(16,8))
ax.bar(lsa_categories, lsa_counts);
ax.set_xticks(lsa_categories);
ax.set_xticklabels(labels);
ax.set_ylabel('Number of review text');
ax.set_title('LSA topic counts');
plt.show();

Figure 30
By looking at the most frequent words in each topic, we have a sense that we may not reach any degree of separation across the topic categories. In another word, we could not separate review text by departments using topic modeling techniques.
Topic modeling techniques have a number of important limitations. To begin, the term “topic” is somewhat ambigious, and by now it is perhaps clear that topic models will not produce highly nuanced classification of texts for our data.
In addition, we can observe that the vast majority of the review text are categorized to the first topic (Topic 0). The t-SNE visualization of LSA topic modeling won’t be pretty.
All the code can be found on the Jupyter notebook. And code plus the interactive visualizations can be viewed on nbviewer.
Happy Monday!
Data Science
NLP
Visualization
Plotly
Python
2.9K claps



Susan Li
WRITTEN BY

Susan Li
Follow
Changing the world, one post at a time. Sr Data Scientist, Toronto Canada. https://www.linkedin.com/in/susanli/
Towards Data Science
Towards Data Science
Follow
A Medium publication sharing concepts, ideas, and codes.
See responses (20)
Discover Medium
Welcome to a place where words matter. On Medium, smart voices and original ideas take center stage - with no ads in sight. Watch
Make Medium yours
Follow all the topics you care about, and we’ll deliver the best stories for you to your homepage and inbox. Explore
Become a member
Get unlimited access to the best stories on Medium — and support writers while you’re at it. Just $5/month. Upgrade
About
Help
Legal

