#!/usr/bin/env python
# coding: utf-8

# In[3]:


# imporitng necessary libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix


# In[4]:


# Load Dataset 
df=pd.read_csv("Churn_Modelling.csv")
df.head()


# In[5]:


#checking for missing values 

print(df.isnull().sum())


# In[6]:


df.info()


# In[7]:


# dropping unnecessary columns
drop_cols = ['CustomerId','Surname','RowNumber']
df.drop(drop_cols, axis=1, inplace=True)
df.head()


# In[8]:


print(df['Geography'].unique())
df['Geography'].value_counts()


# # Exploratory Data Analysis

# In[9]:


# Four subplots for Geography, Gender, Age, and CreditScore distributions


# In[10]:


# Geography Distribution
plt.subplot(2, 2, 1)
sns.countplot(x='Geography', data=df, hue='Exited', palette='Set2')
plt.title('Distribution by Geography')
plt.xlabel('Geography')
plt.ylabel('Count')
plt.legend(title='Churn', loc='upper right')
plt.xticks(rotation=45)


# In[11]:


# Gender Distribution
plt.subplot(2, 2, 2)
sns.countplot(x='Gender', data=df, hue='Exited', palette='Set2')
plt.title('Distribution by Gender')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.legend(title='Churn', loc='upper right')


# In[12]:


# Age Distribution
plt.subplot(2, 2, 3)
sns.histplot(x='Age', data=df, kde=True, hue='Exited', palette='Set2', bins=30)
plt.title('Distribution by Age')
plt.xlabel('Age')
plt.ylabel('Count')


# In[13]:


# CreditScore Distribution
plt.subplot(2, 2, 4)
sns.histplot(x='CreditScore', data=df, kde=True, hue='Exited', palette='Set2', bins=30)
plt.title('Distribution by CreditScore')
plt.xlabel('CreditScore')
plt.ylabel('Count')
plt.tight_layout()
plt.show()


# # Label Encoding 

# In[14]:


label_encoder = LabelEncoder()


# In[15]:


categorical_cols=['Geography','Gender']
for column in categorical_cols:
    df[column] = label_encoder.fit_transform(df[column])    


# In[16]:


# Excited column is the target variable
x = df.drop(columns=['Exited'])
y = df['Exited']


# In[20]:


scaler = StandardScaler()
x = scaler.fit_transform(x)


# In[22]:


print(x.shape)
y.shape


# In[23]:


# Split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


# # Random Forest

# In[24]:


from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
rfc.fit(x_train, y_train)
y_pred = rfc.predict(x_test)

print("Accuracy Score :", accuracy_score(y_test, y_pred)*100,"%")


# # XGBoost

# In[25]:


from xgboost import XGBClassifier
xgb_model = XGBClassifier()
xgb_model.fit(x_train, y_train)
y_pred = xgb_model.predict(x_test)
print("Accuracy Score :", accuracy_score(y_test, y_pred)*100,"%")


# # Logistic Regression
# 

# In[26]:


from sklearn.linear_model import LogisticRegression
lr_model = LogisticRegression()
lr_model.fit(x_train,y_train)
y_pred= lr_model.predict(x_test)
print("Accuracy Score:", accuracy_score(y_test, y_pred) * 100, "%")


# In[27]:


# Visualize Random Forest and XGBoost Algorithm because Random Forest and
# XGBoost Algorithm have the Best Accuracy

y_pred = rfc.predict(x_test)
print("Classification report - n", classification_report(y_test,y_pred))


# In[32]:


# Evaluate the classifier
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

# Display the confusion matrix as a heatmap

plt.figure(figsize=(5, 5))
sns.heatmap(data=conf_matrix, linewidths=.5, annot=True, fmt="d", square=True, cmap='Greens', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
all_sample_title = 'Accuracy Score: {0}'.format(accuracy)
plt.title(all_sample_title, size=15)
plt.show()

