#!/usr/bin/env python
# coding: utf-8

# # Import Libraries

# In[1]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 


# # Load Dataset

# In[2]:


train_data = pd.read_csv('fraudTrain.csv')
test_data = pd.read_csv('fraudTest.csv')


# In[3]:


print("\nTrain Data Sample:")
print(train_data.head())


# In[4]:


# checking for missing values
print("\nMissing values in train data :")
train_data.isnull().sum()


# In[5]:


print("\nMissing values in test data: ")
test_data.isnull().sum()


# In[6]:


# Determine number of fraud cases in dataset

fraud = train_data[train_data['is_fraud'] == 1]
valid = train_data[train_data['is_fraud'] == 0]
outlier_fraction = len(fraud) / float(len(valid)) * 100
print("Fraudulent transactions as a percentage of all transactions:", round(outlier_fraction, 2), "%")
print('Fraud Cases: {}'.format(len(train_data[train_data['is_fraud'] == 1]))) 
print('Valid Transactions: {}'.format(len(train_data[train_data['is_fraud'] == 0]))) 


# In[7]:


print('Amount of transaction details of the fraudulent transaction') 
fraud.amt.describe() 


# In[8]:


# Visualization of Transaction Amount Distribution
 
fraudulent_amount = train_data[train_data['is_fraud'] == 1]['amt'].sum()
valid_amount = train_data[train_data['is_fraud'] == 0]['amt'].sum()

labels = ['Fraudulent', 'Valid']
amounts = [fraudulent_amount, valid_amount]

plt.figure(figsize=(6,6))
plt.pie(amounts, labels=labels,autopct='%1.1f%%',startangle=150)
plt.title('Transaction Amount Distribution')
plt.axis('equal')


# In[9]:


#
#Data Preprocessing: Removing Unnecessary Columns and Handling Missing Values 
#
train_data.drop(columns=['cc_num','first', 'last', 'street', 'city', 'state', 'zip',
       'dob', 'trans_num','trans_date_trans_time'],inplace=True)
train_data.dropna(inplace=True)
#
#
test_data.drop(columns=['cc_num','first', 'last', 'street', 'city', 'state', 'zip',
       'dob', 'trans_num','trans_date_trans_time'],inplace=True)
test_data.dropna(inplace=True)


# # Label Encoding 

# In[10]:


from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()


# In[11]:


categorical_columns = ['merchant', 'category', 'gender', 'job']
for column in categorical_columns:
    train_data[column] = label_encoder.fit_transform(train_data[column])
train_data


# In[12]:


# Data Preparation: Creating Training Data and Target Labels

x_train = train_data.iloc[:,0:11]
x_train


# In[13]:


y_train = train_data['is_fraud']
y_train


# In[14]:


catagorical_columns = ['merchant','category','gender','job']
for column in categorical_columns:
    test_data[column] = label_encoder.fit_transform(test_data[column])
test_data


# In[15]:


x_test = test_data.iloc[:,0:11]
x_test


# In[16]:


y_test = test_data['is_fraud']
y_test


# # Random Forest Model

# In[17]:


# Building the Random Forest Classifier 

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
rfc.fit(x_train, y_train) 
y_Pred = rfc.predict(x_test) 


# In[18]:


from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score,  f1_score,  matthews_corrcoef  
#

n_outliers = len(fraud) 
n_errors = (y_Pred != y_test).sum() 
print("The model used is Random Forest classifier") 

acc = accuracy_score(y_test, y_Pred) 
print("The accuracy is {}".format(acc)) 
  
prec = precision_score(y_test, y_Pred) 
print("The precision is {}".format(prec)) 
  
rec = recall_score(y_test, y_Pred) 
print("The recall is {}".format(rec)) 
  
f1 = f1_score(y_test, y_Pred) 
print("The F1-Score is {}".format(f1)) 
  
MCC = matthews_corrcoef(y_test, y_Pred) 
print("The Matthews correlation coefficient is{}".format(MCC)) 
  


# In[20]:


# Evaluate the classifier
from sklearn.metrics import confusion_matrix
accuracy = accuracy_score(y_test, y_Pred)
conf_matrix = confusion_matrix(y_test, y_Pred)
classification_rep = classification_report(y_test, y_Pred)

# Display the confusion matrix as a heatmap
plt.figure(figsize=(5, 5))
sns.heatmap(data=conf_matrix, linewidths=.5, annot=True, fmt="d", square=True, cmap='Greens', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
all_sample_title = 'Accuracy Score: {0}'.format(accuracy)
plt.title(all_sample_title, size=15)
plt.show()


# In[21]:


from sklearn.linear_model import LogisticRegression
lr_model = LogisticRegression()
lr_model.fit(x_train,y_train)
y_pred= lr_model.predict(x_test)
print("Accuracy Score:", accuracy_score(y_test, y_pred) * 100, "%")


# In[ ]:




