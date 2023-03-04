#!/usr/bin/env python
# coding: utf-8

# # Assignment - 11 (Text Mining)

# ### ONE:
# 1) Perform sentimental analysis on the Elon-musk tweets

# In[143]:


# Install Libraries if not installed
#%pip install spacy
#!python -m spacy download en_core_web_md
#!pip install wordcloud


# In[144]:


get_ipython().run_line_magic('pip', 'install spacy')


# In[145]:


get_ipython().system('python -m spacy download en_core_web_md')


# In[146]:


get_ipython().system('pip install wordcloud')


# In[147]:


get_ipython().system('pip install wordcloud')


# In[148]:


# Import libraries
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import string
import spacy
import nltk
from matplotlib.pyplot import imread
from wordcloud import WordCloud, STOPWORDS


# In[149]:


# load the dataset
Elon=pd.read_csv(r"C:\Users\Lenovo\Desktop\PRITAM_DATA_SCIENCE_ASSIGNMENT\MY_ASSIGNMENT\PritamAssignment11\Question\Elon_musk.csv",encoding='Latin-1')
Elon.drop(['Unnamed: 0'],inplace=True,axis=1)
Elon


# ### Text Preprocessing

# In[150]:


Elon=[Text.strip() for Text in Elon.Text] # remove both the leading and the trailing characters
Elon=[Text for Text in Elon if Text] # removes empty strings, because they are considered in Python as False
Elon[0:10]


# In[151]:


# Joining the list into one string/text
Elon_text=' '.join(Elon)
Elon_text


# In[152]:


# remove Twitter username handles from a given twitter text. (Removes @usernames)
from nltk.tokenize import TweetTokenizer
tknzr = TweetTokenizer(strip_handles=True)
Elon_tokens=tknzr.tokenize(Elon_text)
print(Elon_tokens)


# In[153]:


# Again Joining the list into one string/text
Elon_tokens_text=' '.join(Elon_tokens)
Elon_tokens_text


# In[154]:


# Remove Punctuations 
Punctuations =Elon_tokens_text.translate(str.maketrans('','',string.punctuation))
Punctuations 


# In[155]:


# remove https or url within text
import re
url=re.sub(r'http\S+', '', Punctuations )
url


# In[156]:


#Tokenization
from nltk.tokenize import word_tokenize
nltk.download('punkt')
text_tokens=word_tokenize(url)
print(text_tokens)


# In[157]:


# Tokens count
len(text_tokens)


# In[158]:


# Remove Stopwords
from nltk.corpus import stopwords
nltk.download('stopwords')
my_stop_words=stopwords.words('english')

sw_list = ['\x92','rt','ye','yeah','haha','Yes','U0001F923','I']
my_stop_words.extend(sw_list)

no_stop_tokens=[word for word in text_tokens if not word in my_stop_words]
print(no_stop_tokens)


# In[159]:


# Normalize the data
lower_words=[Text.lower() for Text in no_stop_tokens]
print(lower_words[100:200])


# In[160]:


# Stemming (Optional)
from nltk.stem import PorterStemmer
ps=PorterStemmer()
stemmed_tokens=[ps.stem(word) for word in lower_words]
print(stemmed_tokens[100:200])


# In[161]:


# Lemmatization
get_ipython().system('python -m spacy download en')
nlp=spacy.load('en_core_web_sm')
doc=nlp(' '.join(lower_words))
print(doc)


# In[162]:


lemmas=[token.lemma_ for token in doc]
print(lemmas)


# In[163]:


clean_tweets=' '.join(lemmas)
clean_tweets


# ### Feature Extaction

# #### 1. Using CountVectorizer

# In[164]:


from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
tweetscv=cv.fit_transform(lemmas)


# In[165]:


print(cv.vocabulary_)


# In[166]:


print(cv.get_feature_names()[100:200])


# In[167]:


print(tweetscv.toarray()[100:200])


# In[168]:


print(tweetscv.toarray().shape)


# #### 2. CountVectorizer with N-grams (Bigrams & Trigrams)

# In[169]:


cv_ngram_range=CountVectorizer(analyzer='word',ngram_range=(1,3),max_features=100)
bow_matrix_ngram=cv_ngram_range.fit_transform(lemmas)


# In[170]:


print(cv_ngram_range.get_feature_names())
print(bow_matrix_ngram.toarray())


# #### 3. TF-IDF Vectorizer

# In[171]:


from sklearn.feature_extraction.text import TfidfVectorizer
tfidfv_ngram_max_features=TfidfVectorizer(norm='l2',analyzer='word',ngram_range=(1,3),max_features=500)
tfidf_matix_ngram=tfidfv_ngram_max_features.fit_transform(lemmas)


# In[172]:


print(tfidfv_ngram_max_features.get_feature_names())
print(tfidf_matix_ngram.toarray())


# #### 4. Generate Word Cloud

# In[173]:


# Define a function to plot word cloud
def plot_cloud(wordcloud):
    plt.figure(figsize=(40,30))
    plt.imshow(wordcloud)
    plt.axis('off')
    
# Generate Word Cloud


STOPWORDS.add('pron')
STOPWORDS.add('rt')
STOPWORDS.add('yeah')
wordcloud = WordCloud(width=3000,height=2000,background_color='black',max_words=50,
                   colormap='Set1',stopwords=STOPWORDS).generate(clean_tweets)
plot_cloud(wordcloud)


# #### 5. Named Entity Recognition (NER

# In[174]:


# Parts Of Speech (POS) Tagging
nlp=spacy.load('en_core_web_sm')

one_block=clean_tweets
doc_block=nlp(one_block)
spacy.displacy.render(doc_block,style='ent',jupyter=True)


# In[175]:


for token in doc_block[100:200]:
    print(token,token.pos_)


# In[176]:


# Filtering the nouns and verbs only
nouns_verbs=[token.text for token in doc_block if token.pos_ in ('NOUN','VERB')]
print(nouns_verbs[100:200])


# In[177]:


# Filtering the nouns and verbs only
nouns_verbs=[token.text for token in doc_block if token.pos_ in ('NOUN','VERB')]
print(nouns_verbs[100:200])


# In[178]:


# Counting the noun & verb tokens
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()

X=cv.fit_transform(nouns_verbs)
sum_words=X.sum(axis=0)

words_freq=[(word,sum_words[0,idx]) for word,idx in cv.vocabulary_.items()]
words_freq=sorted(words_freq, key=lambda x: x[1], reverse=True)

wd_df=pd.DataFrame(words_freq)
wd_df.columns=['word','count']
wd_df[0:10] # viewing top ten results


# #### 6. Emotion Mining - Sentiment Analysis

# In[179]:


from nltk import tokenize
sentences=tokenize.sent_tokenize(' '.join(Elon))
sentences


# In[180]:


sent_df=pd.DataFrame(sentences,columns=['sentence'])
sent_df


# In[182]:


afin = pd.read_csv(r"C:\Users\Lenovo\Downloads\Afinn.csv",  encoding='latin-1')
afin
#sep=',',


# In[183]:


affinity_scores=afin.set_index('word')['value'].to_dict()
affinity_scores


# In[184]:


# Custom function: score each word in a sentence in lemmatised form, but calculate the score for the whole original sentence
nlp=spacy.load('en_core_web_sm')
sentiment_lexicon=affinity_scores

def calculate_sentiment(text:str=None):
    sent_score=0
    if text:
        sentence=nlp(text)
        for word in sentence:
            sent_score+=sentiment_lexicon.get(word.lemma_,0)
    return sent_score


# In[185]:


# manual testing
calculate_sentiment(text='great')


# In[186]:


# Calculating sentiment value for each sentence
sent_df['sentiment_value']=sent_df['sentence'].apply(calculate_sentiment)
sent_df['sentiment_value']


# In[187]:


# how many words are there in a sentence?
sent_df['word_count']=sent_df['sentence'].str.split().apply(len)
sent_df['word_count']


# In[188]:


sent_df.sort_values(by='sentiment_value')


# In[189]:


# Sentiment score of the whole review
sent_df['sentiment_value'].describe()


# In[190]:


# negative sentiment score of the whole review
sent_df[sent_df['sentiment_value']<=0]


# In[191]:


# positive sentiment score of the whole review
sent_df[sent_df['sentiment_value']>0]


# In[192]:


# Adding index cloumn
sent_df['index']=range(0,len(sent_df))
sent_df


# In[193]:


# Plotting the sentiment value for whole review
import seaborn as sns
plt.figure(figsize=(15,10))
sns.distplot(sent_df['sentiment_value'])


# In[194]:


# Plotting the line plot for sentiment value of whole review
plt.figure(figsize=(15,10))
sns.lineplot(y='sentiment_value',x='index',data=sent_df)


# In[195]:


# Correlation analysis
sent_df.plot.scatter(x='word_count',y='sentiment_value',figsize=(8,8),title='Sentence sentiment value to sentence word count');


# ### TWO:
# 1) Extract reviews of any product from ecommerce website like amazon
# 2) Perform emotion mining

# In[196]:


#!pip install scrapy


# In[197]:


# Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import string
import spacy

from matplotlib.pyplot import imread
from wordcloud import WordCloud, STOPWORDS
get_ipython().run_line_magic('matplotlib', 'inline')


# In[198]:


# Import extracted amazon reviews Dataset
import pandas as pd
reviews=pd.read_csv(r"C:\Users\Lenovo\Downloads\extract_reviews.csv")
reviews


# ### Text Preprocessing

# In[199]:


reviews=[comment.strip() for comment in reviews.comment] # remove both the leading and the trailing characters
reviews=[comment for comment in reviews if comment] # removes empty strings, because they are considered in Python as False
reviews[0:10]


# In[200]:


# Joining the list into one string/text
reviews_text=' '.join(reviews)
reviews_text


# In[201]:


# Remove Punctuations 
no_punc_text=reviews_text.translate(str.maketrans('','',string.punctuation))
no_punc_text


# In[202]:


# Tokenization
import nltk
nltk.download('punkt')
nltk.download('stopwords')


# In[203]:


from nltk import word_tokenize
text_tokens=word_tokenize(no_punc_text)
print(text_tokens[0:50])


# In[204]:


len(text_tokens)


# In[205]:


# Remove stopwords
from nltk.corpus import stopwords
my_stop_words=stopwords.words('english')

sw_list=['I','The','It','A']
my_stop_words.extend(sw_list)

no_stop_tokens=[word for word in text_tokens if not word in my_stop_words]
print(no_stop_tokens)


# In[206]:


# Normalize the data
lower_words=[comment.lower() for comment in no_stop_tokens]
print(lower_words)


# In[207]:


# Stemming (Optional)
from nltk.stem import PorterStemmer
ps=PorterStemmer()
stemmed_tokens=[ps.stem(word) for word in lower_words]
print(stemmed_tokens)


# In[208]:


# Lemmatization
nlp=spacy.load('en_core_web_sm')
doc=nlp(' '.join(lower_words))
print(doc)


# In[209]:


lemmas=[token.lemma_ for token in doc]
print(lemmas)


# In[210]:


clean_reviews=' '.join(lemmas)
clean_reviews


# ### Feature Extaction

# #### 1. Using CountVectorizer

# In[212]:


from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
reviewscv=cv.fit_transform(lemmas)


# In[213]:


print(cv.vocabulary_)


# In[214]:


print(reviewscv.toarray()[150:300])


# In[215]:


print(reviewscv.toarray().shape)


# #### 2. CountVectorizer with N-grams (Bigrams & Trigrams)

# In[216]:


cv_ngram_range=CountVectorizer(analyzer='word',ngram_range=(1,3),max_features=100)
bow_matrix_ngram=cv_ngram_range.fit_transform(lemmas)


# In[217]:


print(cv_ngram_range.get_feature_names())
print(bow_matrix_ngram.toarray())


# #### 3. TF-IDF Vectorizer

# In[218]:


from sklearn.feature_extraction.text import TfidfVectorizer
tfidfv_ngram_max_features=TfidfVectorizer(norm='l2',analyzer='word',ngram_range=(1,3),max_features=500)
tfidf_matrix_ngram=tfidfv_ngram_max_features.fit_transform(lemmas)


# In[219]:


print(tfidfv_ngram_max_features.get_feature_names())
print(tfidf_matrix_ngram.toarray())


# #### 4. Generate Word Cloud

# In[220]:


# Define a function to plot word cloud
def plot_cloud(wordcloud):
    plt.figure(figsize=(40,30))
    plt.imshow(wordcloud)
    plt.axis('off')

# Generate word cloud

STOPWORDS.add('Pron')
wordcloud=WordCloud(width=3000,height=2000,background_color='white',max_words=100,
                   colormap='Set2',stopwords=STOPWORDS).generate(clean_reviews)
plot_cloud(wordcloud)


# #### 5. Named Entity Recognition (NER)

# In[221]:


# Parts of speech (POS) tagging
nlp=spacy.load('en_core_web_sm')

one_block=clean_reviews
doc_block=nlp(one_block)
spacy.displacy.render(doc_block,style='ent',jupyter=True)


# In[222]:


for token in doc_block[100:200]:
    print(token,token.pos_)


# In[223]:


# Filtering the nouns and verbs only
nouns_verbs=[token.text for token in doc_block if token.pos_ in ('NOUN','VERB')]
print(nouns_verbs[100:200])


# In[224]:


# Counting the noun & verb tokens
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()

X=cv.fit_transform(nouns_verbs)
sum_words=X.sum(axis=0)

words_freq=[(word,sum_words[0,idx]) for word,idx in cv.vocabulary_.items()]
words_freq=sorted(words_freq,key=lambda x: x[1],reverse=True)

wd_df=pd.DataFrame(words_freq)
wd_df.columns=['word','count']
wd_df[0:10] # viewing top ten results


# In[225]:


# Visualizing results (Barchart for top 10 nouns + verbs)
wd_df[0:10].plot.bar(x='word',figsize=(12,8),title='Top 10 nouns and verbs');


# #### 6. Emotion Mining - Sentiment Analysis

# In[226]:


from nltk import tokenize
sentences=tokenize.sent_tokenize(' '.join(reviews))
sentences


# In[227]:


sent_df=pd.DataFrame(sentences,columns=['sentence'])
sent_df


# In[230]:


# Emotion Lexicon - Affin
affin = pd.read_csv(r"C:\Users\Lenovo\Downloads\Afinn.csv" ,sep=',' ,encoding='latin-1')
affin
#sep=',',


# In[231]:


affinity_scores=affin.set_index('word')['value'].to_dict()
affinity_scores


# In[232]:


# Custom function: score each word in a sentence in lemmatised form, but calculate the score for the whole original sentence
nlp=spacy.load('en_core_web_sm')
sentiment_lexicon=affinity_scores

def calculate_sentiment(text:str=None):
    sent_score=0
    if text:
        sentence=nlp(text)
        for word in sentence:
            sent_score+=sentiment_lexicon.get(word.lemma_,0)
    return sent_score


# In[233]:


# manual testing
calculate_sentiment(text='good service')


# In[234]:


# Calculating sentiment value for each sentence
sent_df['sentiment_value']=sent_df['sentence'].apply(calculate_sentiment)
sent_df['sentiment_value']


# In[235]:


# how many words are there in a sentence?
sent_df['word_count']=sent_df['sentence'].str.split().apply(len)
sent_df['word_count']


# In[236]:


sent_df.sort_values(by='sentiment_value')


# In[237]:


# Sentiment score of the whole review
sent_df['sentiment_value'].describe()


# In[238]:


# negative sentiment score of the whole review
sent_df[sent_df['sentiment_value']<=0]


# In[239]:


# positive sentiment score of the whole review
sent_df[sent_df['sentiment_value']>0]


# In[240]:


# Adding index cloumn
sent_df['index']=range(0,len(sent_df))
sent_df


# In[241]:


# Plotting the sentiment value for whole review
import seaborn as sns
plt.figure(figsize=(15,10))
sns.distplot(sent_df['sentiment_value'])


# In[242]:


# Plotting the line plot for sentiment value of whole review
plt.figure(figsize=(15,10))
sns.lineplot(y='sentiment_value',x='index',data=sent_df)

