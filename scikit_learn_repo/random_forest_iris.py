#! python3

'''
Random Forest example with scikit learn lib and iris data
'''

# Libs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Data
iris = datasets.load_iris()
df_iris = pd.DataFrame(iris.data)
df_iris.columns = iris.feature_names
df_iris['target'] = iris.target
iris_dict = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
df_iris['target_name'] = df_iris['target'].map(iris_dict)

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(df_iris.iloc[:, :-2], df_iris.iloc[:, -1], test_size=0.33, random_state=42)

# Call model
clf = RandomForestClassifier(max_depth=2, random_state=0)

# Train the model using the training sets
clf.fit(X_train, y_train)

# Make predictions using the testing set
print(clf.predict([[6, 6, 6, 6]])) # Prediction for a flower measuring 6 cm of sepal length, 6 cm of sepal width, 6 cm of petal length and 6 cm of petal width 
y_pred_test = clf.predict(X_test)

# View accuracy score
accuracy_score(y_test, y_pred_test)
print(f'O modelo apresentou {accuracy_score(y_test, y_pred_test) * 100} % de acurácia')

# View confusion matrix for test data and predictions
confusion_matrix(y_test, y_pred_test)

# Get and reshape confusion matrix data
matrix = confusion_matrix(y_test, y_pred_test)
matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]

# Build the plot
plt.figure(figsize=(16,7))
sns.set(font_scale=1.4)
sns.heatmap(matrix, annot=True, annot_kws={'size':10},
            cmap=plt.cm.Greens, linewidths=0.2)
class_names = iris_dict.values()
tick_marks = iris_dict.keys() + 0.5
plt.xticks(tick_marks, class_names, rotation=0)
plt.yticks(tick_marks, class_names, rotation=0)
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.title('Confusion Matrix for Random Forest Model')
plt.show()

# View the classification report for test data and predictions
print(classification_report(y_test, y_pred_test))


