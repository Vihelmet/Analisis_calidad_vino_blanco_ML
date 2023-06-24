import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import pickle
import yaml

with open(r'./Analisis_calidad_vino_blanco_ML/models/modelo.pkl', 'rb') as archivo_entrada:
    modelo_entrenado = pickle.load(archivo_entrada)

vino_test = pd.read_csv(r'.\Analisis_calidad_vino_blanco_ML\data\test.csv')


X = vino_test.drop(columns=['quality'])
y = vino_test['quality']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=21)

pred_rf_2 = modelo_entrenado.predict(X_test)

print (metrics.classification_report(y_test, pred_rf_2, zero_division=1))