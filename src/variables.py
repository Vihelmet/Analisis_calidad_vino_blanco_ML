import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
import os 

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
print(parent_dir)


vino = pd.read_csv(parent_dir + '/data/raw/winequality-white.csv', delimiter= ';')
vino


vino_features = vino.drop(columns=['quality'])
vino_features

vino_target = vino['quality']

X = vino_features
y = vino_target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=21)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

logisticRegression = LogisticRegression(solver='liblinear', random_state=21)
logisticRegression.fit(X_train, y_train)
pred_logreg = logisticRegression.predict(X_test)

rf = RandomForestClassifier()
rf.fit(X_train, y_train)
pred_rf = rf.predict(X_test)
modelo_arbol = RandomForestClassifier(max_depth=5)