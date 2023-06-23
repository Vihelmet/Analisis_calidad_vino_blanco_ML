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
import pickle

vino_processed = pd.read_csv(r'.\Analisis_calidad_vino_blanco_ML\data\processed.csv', index_col=0)

X = vino_processed.drop(columns=['quality'])
y = vino_processed['quality']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=21)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

logisticRegression = LogisticRegression(solver='liblinear', random_state=21)
logisticRegression.fit(X_train, y_train)

with open(r'./Analisis_calidad_vino_blanco_ML/models/modelo_LogReg.pkl', 'wb') as archivo:
    pickle.dump(logisticRegression, archivo)


rf = RandomForestClassifier()
rf.fit(X_train, y_train)

with open(r'./Analisis_calidad_vino_blanco_ML/models/modelo_RandForClas.pkl', 'wb') as archivo:
    pickle.dump(rf, archivo)