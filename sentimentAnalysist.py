# -*- coding: utf-8 -*-
"""
@author: samim

"""

import pandas as pd
import numpy as np
column = ['yorum']
df = pd.read_csv('urun_yorumları.csv', encoding ='iso-8859-9', sep='"')
df.columns=column
df.info()
df.head()

def remove_stopwords(df_fon):
    stopwords = open('turkce-stop-words', 'r').read().split()
    df_fon['stopwords_removed'] = list(map(lambda doc:
        [word for word in doc if word not in stopwords], df_fon['yorum']))

remove_stopwords(df)

df['Positivity'] = 1

df.Positivity.iloc[4192:] = 0

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df['yorum'], df['Positivity'], random_state = 0)
print(X_train.head())
print('\n\nX_train shape: ', X_train.shape)

from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer(encoding ='iso-8859-9').fit(X_train)

X_train_vectorized = vect.transform(X_train) 

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train_vectorized, y_train)

from sklearn.metrics import roc_auc_score
predictions = model.predict(vect.transform(X_test))
print('AUC: ', roc_auc_score(y_test, predictions))

feature_names = np.array(vect.get_feature_names())
sorted_coef_index = model.coef_[0].argsort()
print('Negatif Kelimeler: \n{}\n'.format(feature_names[sorted_coef_index[:10]]))
print('Pozitif Kelimeler: \n{}\n'.format(feature_names[sorted_coef_index[:-11:-1]]))


from sklearn.feature_extraction.text import TfidfVectorizer
vect = TfidfVectorizer(min_df = 5).fit(X_train)

X_train_vectorized = vect.transform(X_train)
model = LogisticRegression()
model.fit(X_train_vectorized, y_train)
predictions = model.predict(vect.transform(X_test))
print('AUC: ', roc_auc_score(y_test, predictions))  

feature_names = np.array(vect.get_feature_names())
sorted_tfidf_index = X_train_vectorized.max(0).toarray()[0].argsort()
print('En küçük Tf-idf: \n{}\n'.format(feature_names[sorted_tfidf_index[:10]]))
print('En büyük Tf-idf: \n{}\n'.format(feature_names[sorted_tfidf_index[:-11:-1]]))


vect = CountVectorizer(min_df = 5, ngram_range = (1,2)).fit(X_train)
X_train_vectorized = vect.transform(X_train)


model = LogisticRegression()
model.fit(X_train_vectorized, y_train)
predictions = model.predict(vect.transform(X_test))
print('AUC: ', roc_auc_score(y_test, predictions))

feature_names = np.array(vect.get_feature_names())
sorted_coef_index = model.coef_[0].argsort()
print('Negatif Olarak Bilinen 10 Kelime: \n{}\n'.format(feature_names[sorted_coef_index][:10]))
print('Pozitif Olarak Bilinen 10 Kelime: \n{}\n'.format(feature_names[sorted_coef_index][:-11:-1]))

while(True):
    yorum=input("Lütfen bir yorum girin:")
    if(yorum == 'X' or yorum == 'x'):
        break
    else:
        print(model.predict(vect.transform([yorum])))