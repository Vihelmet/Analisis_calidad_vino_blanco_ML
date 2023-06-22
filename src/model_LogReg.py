from variables import *
from data_processing import *

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

X = vino_features
y = vino_target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=21)

print((len(X_train), len(X_test)))
print((len(y_train), len(y_test)))


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

logisticRegression = LogisticRegression(solver='liblinear', random_state=21)
logisticRegression.fit(X_train, y_train)

pred_logreg = logisticRegression.predict(X_test)
print(classification_report(y_test, pred_logreg, zero_division=1))