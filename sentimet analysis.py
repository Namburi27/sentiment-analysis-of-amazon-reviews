#!/usr/bin/env python
# coding: utf-8

# In[33]:


import warnings
warnings.filterwarnings('ignore')
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
from sklearn.metrics import accuracy_score


# In[34]:


data = pd.read_csv('AmazonReview.csv')
data.head()


# In[35]:


data.info()


# In[36]:


data.dropna(inplace=True)


# In[37]:


df.drop("Unnamed: 0",inplace=True,axis=1)


# In[38]:


data.loc[data['Sentiment']<=3,'Sentiment'] = 0


# In[39]:


data.loc[data['Sentiment']>3,'Sentiment'] = 1


# In[40]:


stp_words=stopwords.words('english')
def clean_review(review): 
  cleanreview=" ".join(word for word in review.
                       split() if word not in stp_words)
  return cleanreview 
 
data['Review']=data['Review'].apply(clean_review)


# In[41]:


data.head()


# In[42]:


data['Sentiment'].value_counts()


# In[43]:


consolidated=' '.join(word for word in data['Review'][data['Sentiment']==0].astype(str))
wordCloud=WordCloud(width=1600,height=800,random_state=21,max_font_size=110)
plt.figure(figsize=(15,10))
plt.imshow(wordCloud.generate(consolidated),interpolation='bilinear')
plt.axis('off')
plt.show()


# In[44]:


consolidated=' '.join(word for word in data['Review'][data['Sentiment']==1].astype(str))
wordCloud=WordCloud(width=1600,height=800,random_state=21,max_font_size=110)
plt.figure(figsize=(15,10))
plt.imshow(wordCloud.generate(consolidated),interpolation='bilinear')
plt.axis('off')
plt.show()


# In[45]:


cv = TfidfVectorizer(max_features=2500)
X = cv.fit_transform(data['Review'] ).toarray()


# In[58]:


from sklearn.model_selection import train_test_split
x_train ,x_test,y_train,y_test=train_test_split(X,data['Sentiment'],
                                                test_size=0.30 ,
                                                random_state=42)


# In[59]:


from sklearn.linear_model import LogisticRegression


# In[ ]:


model=LogisticRegression()
model.fit(x_train,y_train)


# In[ ]:


pred=model.predict(x_test)


# In[57]:


print(accuracy_score(y_test,pred))


# In[52]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,pred)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = cm, 
                                            display_labels = [False, True])
cm_display.plot()
plt.show()


# In[ ]:




