# Phishing-Email-Detection

## Problem Statement
> Given a dataset of phishing & normal emails, our task is to detect if a given email is a phishing email or not using a ML-led solution.

## Dataset
• There are 2 folders: ‘Normal’ and ‘Phishing’.

• ‘Normal’ contains all the legitimate emails. There are 2551 normal mails.

• ‘Phishing’ contains phishing emails. There are 2274 such emails.

## Approach

• Our dataset has .eml files. To extract body part of emails I have used email parser. This parser can understand email document structure. We can get much information from this parser but I have extract only body part.

• To parse only text from body part I have used beautifulsoup library. This library can help us to get text from html document.

• Used some pre-processing techniques using nltk library. Main purpose of performing this step was to get most frequent words. I have performed lowering text, removal of punctuation, removal of stop words and lemmatization.

• After completing previous step, I got to know some of the features which might help us to predict our target variable. So, I have extracted these features from email’s body part.

• I have performed previously mentioned steps on both of the folders (‘normal’ & ‘phishing’) to generate dataset for each folder.

• Merged both of the datasets for visualisation purpose and some pre-processing tasks. Then applied various machine learning algorithms to predict output class.

• Here are algorithms which I have used:
1) Logistic Regression
2) Support Vector Machine
3) Decision Tree Classifier
4) Random Forest Classifier
5) Bernoulli Naïve Bayes 

• Split data into 80:20 ratio. Then used gridsearchcv to find the optimal hyperparameters of a model which results in the most accurate predictions. Then chose best model.

## Results

1. True Positive Rate: The percentage of phishing emails in the training data set that were correctly classified by the algorithm.

  TPR = 99.55

2. True Negative Rate: The percentage of legitimate emails that were correctly classified as legitimate by the algorithm.

  TNR = 89.60

3. False Positive Rate: It is the percentage of legitimate emails that were incorrectly classified by the algorithm as phishing emails.

  FPR = 10.39

4. False Negative Rate: The number of phishing emails that were incorrectly classified as legitimate by the algorithm.

  FNR = 0.79

5. Precision: Measures the exactness of the classifier. i.e., what percentage of emails that the classifier labelled as phishing are actually phishing emails.

  Precision = TP / (TP + FP) = 89.46

6. Recall: Measures the completeness of the classifier results. i.e., what percentage of phishing emails did the classifier label as phishing.

  Recall = TP / (TP + FN) = 99.55


