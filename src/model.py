import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn import tree
import pickle
import yaml

train = pd.read_csv(r'.\Analisis_calidad_vino_blanco_ML\data\train.csv', index_col=0)

X = train.drop(columns=['good quality'])
y = train['good quality']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=21)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

rf = RandomForestClassifier(n_estimators= 100, random_state=21)
rf.fit(X_train, y_train)

# pred_rf = rf.predict(X_test)
# print(rf.score(X_test, y_test))

with open(r'./Analisis_calidad_vino_blanco_ML/models/modelo_rf_def.pkl', 'wb') as archivo:
    pickle.dump(rf, archivo)

with open(r'./Analisis_calidad_vino_blanco_ML/models/model_config_def.yaml', 'w') as c:
    yaml.dump(rf, c)