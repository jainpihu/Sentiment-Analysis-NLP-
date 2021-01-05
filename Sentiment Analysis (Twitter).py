#!/usr/bin/env python
# coding: utf-8

# In[2]:


import re
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
import string
import nltk
import warnings 
warnings.filterwarnings("ignore", category=DeprecationWarning)

get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


train  = pd.read_csv('train_E6oV3lV.csv')
test = pd.read_csv('test_tweets_anuFYb8.csv')


# In[4]:


train.head()


# In[5]:


test.head()


# In[6]:


combi = train.append(test, ignore_index=True)


# In[7]:


combi.head()


# In[8]:


def remove_pattern(input_text,pattern):
    r=re.findall(pattern,input_text)
    for i in r:
        input_text=re.sub(i,"",input_text)
    return input_text


# In[9]:


combi["tidy_tweets"]=np.vectorize(remove_pattern)(combi["tweet"],"@[\w]*")


# In[10]:


combi.head()


# In[11]:


combi['tidy_tweets'] = combi['tidy_tweets'].str.replace("[^a-zA-Z#]", " ")


# In[12]:


combi.head()


# In[13]:


combi['tidy_tweets'] = combi['tidy_tweets'].apply(lambda x: " ".join([w for w in x.split() if len(w)>3]))


# In[14]:


combi.head()


# In[15]:


tokenized_tweet=combi["tidy_tweets"].apply(lambda x: x.split())


# In[16]:


tokenized_tweet.head()


# In[17]:


from nltk.stem.porter import *
stemmer=PorterStemmer()

tokenized_tweet=tokenized_tweet.apply(lambda x: [stemmer.stem(i) for i in x])


# In[18]:


tokenized_tweet.head()


# In[20]:


for i in range(len(tokenized_tweet)):
    tokenized_tweet[i] = ' '.join(tokenized_tweet[i])

combi['tidy_tweets'] = tokenized_tweet


# In[28]:


combi.head(20)


# In[22]:


tokenized_tweet.head()


# In[24]:


get_ipython().system('pip install wordcloud')


# In[25]:


all_words=" ".join(text for text in combi['tidy_tweets'])
from wordcloud import WordCloud
wordcloud=WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(all_words)

plt.figure(figsize=(10,7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()


# In[27]:


normal_words=" ".join(text for text in combi['tidy_tweets'][combi['label']==0])
from wordcloud import WordCloud
wordcloud=WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(normal_words)

plt.figure(figsize=(10,7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()


# In[30]:


negative_words=" ".join(text for text in combi['tidy_tweets'][combi['label']==1])
from wordcloud import WordCloud
wordcloud=WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(negative_words)

plt.figure(figsize=(10,7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()


# In[38]:


def hashtag_extract(xht_
    hashtags=[]
    for text in x:
        ht = re.findall(r"#(\w+)", text)
        hashtags.append(ht)
    return hashtags


# In[37]:


ht_regular=hashtag_extract(combi['tidy_tweets'][combi['label']==0])

ht_negative=hashtag_extract(combi['tidy_tweets'][combi['label']==1])

ht_regular=sum(ht_regular,[])
ht_negative=sum(ht_negative,[])


# In[46]:


a=nltk.FreqDist(ht_regular)
d=pd.DataFrame({'Hashtags':list(a.keys()),
               'Count':list(a.values())})
d=d.nlargest(columns='Count',n=10)
plt.figure(figsize=(16,5))
ax=sns.barplot(data=d,x='Hashtags',y='Count')
ax.set(ylabel = 'Count')
plt.show()


# In[47]:


from sklearn.feature_extraction.text import CountVectorizer
bow_vectorizer=CountVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')

bow=bow_vectorizer.fit_transform(combi['tidy_tweets'])


# In[48]:


from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')
# TF-IDF feature matrix
tfidf = tfidf_vectorizer.fit_transform(combi['tidy_tweets'])


# In[52]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

train_bow = bow[:31962,:]
test_bow = bow[31962:,:]

# splitting data into training and validation set
xtrain_bow, xvalid_bow, ytrain, yvalid = train_test_split(train_bow, train['label'], random_state=42, test_size=0.3)

lreg = LogisticRegression()
lreg.fit(xtrain_bow, ytrain) # training the model

prediction = lreg.predict_proba(xvalid_bow) # predicting on the validation set
prediction_int = prediction[:,1] >= 0.3 # if prediction is greater than or equal to 0.3 than 1 else 0
prediction_int = prediction_int.astype(np.int)

print(f1_score(yvalid, prediction_int)) # calculating f1 score


# In[54]:


test_pred = lreg.predict_proba(test_bow)
test_pred_int = test_pred[:,1] >= 0.3
test_pred_int = test_pred_int.astype(np.int)
test['label'] = test_pred_int
test.head()


# In[ ]:




