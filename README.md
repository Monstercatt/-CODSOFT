# -CODSOFT
Task1 summary: Credit Card  Fraud Detection with Random Forest Classifier

This Python script focuses on fraud detection using a Random Forest Classifier. The main steps are outlined below:

Data Loading and Exploration:

    Utilizes the Pandas library to load training and test data from CSV files.
    Displays a sample of the training data and checks for missing values.
Data Analysis:
  
    Determines the percentage of fraudulent transactions in the training data.
    Provides summary statistics for the amount of fraudulent transactions.
    Visualizes the distribution of transaction amounts between fraudulent and valid transactions using a pie chart.
Data Preprocessing:

    Removes unnecessary columns and handles missing values in both training and test datasets.
Label Encoding:

    Utilizes Scikit-learn's LabelEncoder to convert categorical variables ('merchant', 'category', 'gender', 'job') into        numerical format for both training and test datasets.
Data Preparation:

    Separates the features ('x_train' and 'x_test') and target labels ('y_train' and 'y_test').
Model Building:

    Implements a Random Forest Classifier using Scikit-learn.
    Trains the classifier on the training data and makes predictions on the test data.
Model Evaluation:

    Calculates and prints various performance metrics such as accuracy, precision, recall, F1-score, and Matthews               correlation coefficient.
    Utilizes a confusion matrix and displays it as a heatmap using Seaborn.
Results Visualization:

    Presents the accuracy score on the confusion matrix heatmap.

Task 2: Code Summary: Churn Prediction Analysis

This Python script performs churn prediction analysis using machine learning algorithms. The key steps and findings are as follows:

Data Loading and Exploration:

    Loads the dataset from "Churn_Modelling.csv" and inspects the first few rows.
    Checks for missing values, revealing a clean dataset with no null entries.
    Drops unnecessary columns such as 'CustomerId,' 'Surname,' and 'RowNumber.'
    Exploratory Data Analysis (EDA):

    Visualizes the distribution of churn across different categories, including Geography, Gender, Age, and CreditScore.
    Provides insights into how these factors may influence customer churn.
Label Encoding:

    Converts categorical columns ('Geography' and 'Gender') into numerical format using Label Encoding.
Data Preprocessing:

    Splits the dataset into features ('x') and the target variable ('y').
    Standardizes the feature set using StandardScaler.
    Splits the data into training and testing sets.
Modeling:

    Utilizes three machine learning models: Random Forest Classifier, XGBoost Classifier, and Logistic Regression.
    Trains each model on the training data and evaluates their accuracy on the test set.
Model Comparison:

    Compares the accuracy scores of Random Forest, XGBoost, and Logistic Regression models.
    Random Forest achieves the highest accuracy (86.8%), followed by XGBoost (86.1%) and Logistic Regression (81.5%).
Model Evaluation:

    Provides a classification report for the Random Forest model, detailing precision, recall, and F1-score for both            classes.
    Visualizes the confusion matrix as a heatmap, showcasing the model's performance on the test data.
    This script serves as a comprehensive analysis for predicting churn in a given dataset, demonstrating the application of multiple machine learning algorithms and their respective performance metrics.



Task 3 : Spam Detection with Naive Bayes

This Python script focuses on spam detection using the Naive Bayes classifier with TF-IDF vectorization. The key steps and findings are summarized below:

Importing Libraries:

    Imports necessary libraries such as Pandas, Seaborn, Matplotlib, NLTK, and Scikit-learn.
Loading Dataset:

    Reads a dataset from "spam.csv" and explores its structure, confirming the absence of null values.
Label Encoding and Feature Engineering:

    Renames columns, encodes labels ('ham' as 0, 'spam' as 1), and adds features like message length, word count, and          sentence count.
    Visualizes the distribution of spam and ham messages.
Text Preprocessing:

    Converts text to lowercase and removes punctuation and stopwords.
    Tokenizes the text and preprocesses it using NLTK functions.
Data Splitting and TF-IDF Vectorization:

    Splits the dataset into training and testing sets.
    Applies TF-IDF vectorization to convert text data into numerical features.
Model Training and Evaluation:

    Utilizes the Multinomial Naive Bayes classifier to train on TF-IDF vectors.
    Evaluates the model's accuracy, confusion matrix, and classification report on the test set.
Results Visualization:

    Displays a heatmap of the confusion matrix for a visual representation of model performance.
Conclusion:

    Achieves a high accuracy of approximately 97.3% in spam detection.
    The confusion matrix and classification report provide insights into precision, recall, and F1-score for both spam and     ham classes.










