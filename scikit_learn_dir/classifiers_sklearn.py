from operator import concat
import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

# Wine quality dataset
# https://archive.ics.uci.edu/ml/datasets/wine
df = pd.read_csv('data/winequality-red.csv', sep=';')
X = df.iloc[:, :-2]
y = df.iloc[:, -1]

# Scale transformation
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# KNN
clf_KNN=KNeighborsClassifier(n_neighbors=5)
clf_KNN.fit(X_train, y_train)
clf_KNN_score = [clf_KNN.score(X_test, y_test), "KNN"]; clf_KNN_score
y_pred_KNN_test = clf_KNN.predict(X_test)

# Arvore de decisao
clf_arvore=DecisionTreeClassifier()
clf_arvore.fit(X_train, y_train)
clf_arvore_score = [clf_arvore.score(X_test, y_test), "Arvore"]; clf_arvore_score
y_pred_arvore_test = clf_arvore.predict(X_test)

# Random Forest
clf_floresta=RandomForestClassifier(max_depth=10,random_state=1)
clf_floresta.fit(X_train, y_train)
clf_floresta_score = [clf_floresta.score(X_test, y_test), "Random Forest"]; clf_floresta_score
y_pred_floresta_test = clf_floresta.predict(X_test)

# SVM
clf_svm=SVC(gamma='auto',kernel='rbf')
clf_svm.fit(X_train, y_train)
clf_svm_score = [clf_svm.score(X_test, y_test), "SVM"]; clf_svm_score
y_pred_svm_test = clf_svm.predict(X_test)

# Multi-layer Perceptron
clf_mlp=MLPClassifier(alpha=1e-5,hidden_layer_sizes=(5,5),random_state=1)
clf_mlp.fit(X_train, y_train)
clf_mlp_score = [clf_mlp.score(X_test, y_test), "Multi-layer Perceptron"]
y_pred_mlp_test = clf_mlp.predict(X_test)

# Accuracy matrix
scores_df = pd.DataFrame([clf_KNN_score, clf_arvore_score, clf_floresta_score, clf_svm_score, clf_mlp_score])
scores_df = scores_df.rename(columns={0: "accuracy", 1: "algorithms"}).sort_values('accuracy', ascending=False)
[ "The algorithm " + str(scores_df.iloc[0, 1]) + " scored " + str(scores_df.iloc[0, 0]*100) + "%" + " accuracy"]


