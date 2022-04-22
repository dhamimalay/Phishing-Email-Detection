import re
import os
import glob
import pandas as pd
import numpy as np
import seaborn as sns
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV

import warnings
warnings.filterwarnings('ignore')


def get_df(path, target):

    import email

    for file in os.listdir(path):
        file_loc = os.path.join(path, file)
        base = os.path.splitext(file_loc)[0]
        os.rename(file_loc, base + ".eml")

    path = path + '\*.eml'
    mail_body = None
    email_body = []
    html_tag = []
    img_tag = []
    files = glob.glob(path)

    for file in files:

        with open(file, errors="ignore") as f:
            obj = email.message_from_string(f.read())

        if obj.is_multipart():
            for payload in obj.get_payload():
                mail_body = payload.get_payload()
        else:
            mail_body = obj.get_payload()

        try:
            soup = BeautifulSoup(mail_body, 'html.parser')
            body = re.sub(r'\s+', ' ', soup.get_text()).strip()
            email_body.append(body)
            html_tag.append(len(soup.findAll()))
            img_tag.append(len(soup.find_all('img')))

        except:
            pass

    url_count = []
    dot_count = []

    account = []
    ebay = []
    paypal = []
    email_ = []
    please = []
    information = []
    message = []
    bank = []
    policy = []
    access = []
    member = []
    update = []

    for body in email_body:

        if 'account' in body.lower():
            account.append(1)
        else:
            account.append(0)

        if 'ebay' in body.lower():
            ebay.append(1)
        else:
            ebay.append(0)

        if 'paypal' in body.lower():
            paypal.append(1)
        else:
            paypal.append(0)

        if 'email' in body.lower():
            email_.append(1)
        else:
            email_.append(0)

        if 'please' in body.lower():
            please.append(1)
        else:
            please.append(0)

        if 'information' in body.lower():
            information.append(1)
        else:
            information.append(0)

        if 'message' in body.lower():
            message.append(1)
        else:
            message.append(0)

        if 'bank' in body.lower():
            bank.append(1)
        else:
            bank.append(0)

        if 'policy' in body.lower():
            policy.append(1)
        else:
            policy.append(0)

        if 'access' in body.lower():
            access.append(1)
        else:
            access.append(0)

        if 'member' in body.lower():
            member.append(1)
        else:
            member.append(0)

        if 'update' in body.lower():
            update.append(1)
        else:
            update.append(0)

        pattern = r"\b((?:https?://)?(?:(?:www\.)?(?:[\da-z\.-]+)\.(?:[a-z]{2,6})|(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)|(?:(?:[0-9a-fA-F]{1,4}:){7,7}[0-9a-fA-F]{1,4}|(?:[0-9a-fA-F]{1,4}:){1,7}:|(?:[0-9a-fA-F]{1,4}:){1,6}:[0-9a-fA-F]{1,4}|(?:[0-9a-fA-F]{1,4}:){1,5}(?::[0-9a-fA-F]{1,4}){1,2}|(?:[0-9a-fA-F]{1,4}:){1,4}(?::[0-9a-fA-F]{1,4}){1,3}|(?:[0-9a-fA-F]{1,4}:){1,3}(?::[0-9a-fA-F]{1,4}){1,4}|(?:[0-9a-fA-F]{1,4}:){1,2}(?::[0-9a-fA-F]{1,4}){1,5}|[0-9a-fA-F]{1,4}:(?:(?::[0-9a-fA-F]{1,4}){1,6})|:(?:(?::[0-9a-fA-F]{1,4}){1,7}|:)|fe80:(?::[0-9a-fA-F]{0,4}){0,4}%[0-9a-zA-Z]{1,}|::(?:ffff(?::0{1,4}){0,1}:){0,1}(?:(?:25[0-5]|(?:2[0-4]|1{0,1}[0-9]){0,1}[0-9])\.){3,3}(?:25[0-5]|(?:2[0-4]|1{0,1}[0-9]){0,1}[0-9])|(?:[0-9a-fA-F]{1,4}:){1,4}:(?:(?:25[0-5]|(?:2[0-4]|1{0,1}[0-9]){0,1}[0-9])\.){3,3}(?:25[0-5]|(?:2[0-4]|1{0,1}[0-9]){0,1}[0-9])))(?::[0-9]{1,4}|[1-5][0-9]{4}|6[0-4][0-9]{3}|65[0-4][0-9]{2}|655[0-2][0-9]|6553[0-5])?(?:/[\w\.-]*)*/?)\b"
        url_matches = re.findall(pattern, body)
        url_count.append(len(url_matches))

        max_dot_count = 0
        for url in url_matches:
            dots = url.count('.')
            if max_dot_count < dots:
                max_dot_count = dots
        dot_count.append(max_dot_count)

    df = pd.DataFrame()

    df['HTML_tag'] = html_tag
    df['img_tag'] = img_tag
    df['dots'] = dot_count
    df['urls'] = url_count
    df['account'] = account
    df['ebay'] = ebay
    df['paypal'] = paypal
    df['email'] = email_
    df['please'] = please
    df['information'] = information
    df['message'] = message
    df['bank'] = bank
    df['policy'] = policy
    df['access'] = access
    df['member'] = member
    df['update'] = update
    df['target'] = target

    return df


if __name__ == "__main__":

    print('Processing.......')

    phishing_path = r'C:\Users\Hp\Canary Mail\Phishing Dataset\phishing'
    phishing_df = get_df(phishing_path, 'Phishing')

    normal_path = r'C:\Users\Hp\Canary Mail\Phishing Dataset\normal'
    normal_df = get_df(normal_path, 'Normal')

    df = pd.concat([normal_df, phishing_df], axis=0)
    df = df.sample(frac=1.0, random_state=21).reset_index(drop=True)
    df = df.drop('message', axis=1)

    df['urls'] = np.where(df['urls']>20, 20, df['urls'])
    df['HTML_tag'] = np.where(df['HTML_tag']>50, 50, df['HTML_tag'])
    df['img_tag'] = np.where(df['img_tag']==0, 0, 1)
    df['target'] = np.where(df['target'] == 'Phishing', 1, 0)

    X = df.drop('target', axis=1)
    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.80, stratify=y, random_state=1)

    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)


    def best_estimator(model_name, model, params):

        grid = GridSearchCV(estimator=model, param_grid=params, cv=10, scoring='accuracy', verbose=False, n_jobs=-1)
        grid.fit(X_train_scaled, y_train)

        best_model = grid.best_estimator_

        test_pred = best_model.predict(X_test_scaled)
        train_score = best_model.score(X_train_scaled, y_train)
        test_score = best_model.score(X_test_scaled, y_test)

        print('>       Best Parameters:', grid.best_params_)
        print('>            Best Score:', round(grid.best_score_, 4))
        print('> Training set accuracy:', round(train_score, 4))
        print('>  Testing set accuracy:', round(test_score, 4))

        cf_matrix = confusion_matrix(y_test, test_pred)
        ax = sns.heatmap(cf_matrix, annot=True, fmt='', cmap='Blues')
        ax.set_xlabel('\nPredicted Values', fontsize=12)
        ax.set_ylabel('Actual Values\n', fontsize=12)
        ax.set_title('Confusion Matrix :: {} (Testing Set)\n'.format(model_name), fontsize=15)
        ax.xaxis.set_ticklabels(['Normal', 'Phishing'])
        ax.yaxis.set_ticklabels(['Normal', 'Phishing'])
        plt.show()

        return best_model


    models = {
        'Logistic Regression': LogisticRegression(),
        'SVC': SVC(),
        'Decision Tree': DecisionTreeClassifier(),
        'Random Forest Classifier': RandomForestClassifier(random_state=21),
        'Bernoulli Naive Bayes': BernoulliNB()
    }

    rfc_params = {
        'n_estimators': [40, 50, 60],
        'max_depth': [6, 8, 10, 12]
    }

    lr_params = {
        'solver': ['liblinear'],
        'penalty': ['l1', 'l2'],
        'C': np.logspace(-3, 3, 7)
    }

    svc_params = {
        'C': [2, 3, 4, 5, 10],
        'kernel': ['poly', 'rbf', 'sigmoid'],
        'gamma': [0.1, 0.5, 1, 1.5, 2]
    }

    dt_params = {
        'criterion': ['gini', 'entropy'],
        'max_depth': [6, 10, 14, 16],
        'max_leaf_nodes': [2, 5, 8, 10, 15]
    }

    bnb_params = {
        'alpha': [1]
    }

    model_params = {
        'Logistic Regression': lr_params,
        'SVC': svc_params,
        'Decision Tree': dt_params,
        'Random Forest Classifier': rfc_params,
        'Bernoulli Naive Bayes': bnb_params
    }

    best_models = []
    for model_name, model in models.items():
        print('-' * 80)
        print('{:^81}'.format(model_name))
        print('-' * 80)
        best_models.append(best_estimator(model_name, model, model_params[model_name]))
        print()

    print('-' * 80)
    final_model = best_models[3]
    print('Final Model:', final_model)
    print('-' * 80)