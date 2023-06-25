import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn import tree
import pickle
import yaml

with open(r'./Analisis_calidad_vino_blanco_ML/models/modelo.pkl', 'rb') as archivo_entrada:
    modelo_entrenado = pickle.load(archivo_entrada)

vino_test = pd.read_csv(r'.\Analisis_calidad_vino_blanco_ML\data\test.csv', index_col=0)


X = vino_test.drop(columns=['quality', 'good quality'])
y = vino_test['good quality']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=21)
rf = RandomForestClassifier(n_estimators= 100, random_state=21)
rf.fit(X_test, y_test)
pred_rf = modelo_entrenado.predict(X_test)
print(rf.score(X_test, y_test))
print(metrics.classification_report(y_test, pred_rf, zero_division=1))
print("MAE: ", mean_absolute_error(y_test, pred_rf))
