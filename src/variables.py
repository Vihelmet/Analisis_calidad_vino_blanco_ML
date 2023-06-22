import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import tree

vino = pd.read_csv('../data/raw/winequality-white.csv', delimiter= ';')
vino

vino_features = vino.drop(columns=['fixed acidity', 'citric acid', 'residual sugar', 'free sulfur dioxide', 'total sulfur dioxide', 'pH', 'sulphates','quality'])
vino_features

vino_target = vino['quality']

X_train, X_test, y_train, y_test = train_test_split(vino_features, vino_target, test_size=0.20, random_state=21)

logisticRegression = LogisticRegression(solver='liblinear')
logisticRegression.fit(X_train, y_train)

decisionTree = tree.DecisionTreeClassifier()
decisionTree.fit(X_train, y_train)