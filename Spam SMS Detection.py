#!/usr/bin/env python
# coding: utf-8

# In[130]:


# imporitng necessary libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import nltk
from nltk.corpus import stopwords 
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize

from  sklearn.feature_extraction.text import CountVectorizer
from  sklearn.feature_extraction.text import TfidfTransformer 
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# # Load Dataset

# In[ ]:


# load Dataset 
path="spam.csv"
file=pd.read_csv(path, encoding = "ANSI",usecols=["v1","v2"])
print(file.head(2))


# In[132]:


file.info()


# In[133]:


file.describe()


# In[134]:


# Check Null Values
file.isnull().sum()


# # Label Encoder

# In[135]:


file = file.rename(columns={'v1':'label','v2':'Text'})
file['label_enc'] = file['label'].map({'ham':0,'spam':1})
file["Length"]= file["Text"].apply(len)
file.head()


# In[136]:


# Data Visualization ham vs spam 
sns.countplot(x=file['label'],palette='Set2')
plt.show()


# In[137]:


# length of the message can help us to have more freatures in our model
sns.distplot(file["Length"],bins=30,color='purple')


# In[138]:


# Adding Features  

# Total No. of Words in Data
nltk.download('punkt')
file["word"] = file["Text"].apply(lambda x:len( nltk.word_tokenize(x)))

# Total No. of Sentence
file["sentence"] = file["Text"].apply(lambda x:len(nltk.sent_tokenize(x)))


# In[139]:


file.head(2)


# In[140]:


# Each scatterplot in the grid shows the relationship between two variables
sns.pairplot(file,hue="label")


# # Basic preprocessing for common NLP tasks includes converting text to lowercase and removing punctuation and stopwords.
# #Further steps, especially for text classification tasks, are:
# 
# 1)Tokenization    2)Vectorization and   3)TF-IDF weighting

# In[141]:


# Convert text to lowercase
file['Text'] = file['Text'].str.lower()


# In[142]:


# Remove punctuation and stopwords

nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    words = word_tokenize(text)
    words = [word for word in words if word.isalnum() and word not in stop_words]
    return ' '.join(words)

file['Text'] = file['Text'].apply(preprocess_text)



# In[149]:


# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(file['Text'], file['label'], test_size=0.2, random_state=42)


# TF-IDF Vectorization
tfidf_vectorizer = TfidfVectorizer(max_features=3000)  # Adjust max_features as needed
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)


# In[150]:


print("\nPreprocessed Data:\n")
print(file)

print("\nTF-IDF Vectors:\n")
print(X_train_tfidf.toarray())


# In[151]:


# Train a Naive Bayes classifier
nb_classifier = MultinomialNB()
nb_classifier.fit(X_train_tfidf, y_train)
y_pred = nb_classifier.predict(X_test_tfidf)


# In[153]:


# Evaluate the classifier
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)
print("Accuracy:", accuracy)
print("\nConfusion Matrix:\n", conf_matrix)
print("\nClassification Report:\n", classification_rep)


# In[154]:


# Display the confusion matrix as a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()



# In[ ]:




