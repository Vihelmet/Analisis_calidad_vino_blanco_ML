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

vino_processed = pd.read_csv(r'.\Analisis_calidad_vino_blanco_ML\data\processed.csv', index_col=0)

X = vino_processed.drop(columns=['quality'])
y = vino_processed['quality']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=21)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

rf = RandomForestClassifier(n_estimators= 100, random_state=21)
rf.fit(X_train, y_train)

n_estimators = np.random.uniform(70, 80, 5).astype(int)
max_features = np.random.normal(6, 3, 5).astype(int)

max_features[max_features <= 0] = 1
max_features[max_features > X.shape[1]] = X.shape[1]

hyperparameters = {'n_estimators': list(n_estimators),
                   'max_features': list(max_features)}

print (hyperparameters)

randomCV = RandomizedSearchCV(RandomForestClassifier(), param_distributions=hyperparameters, n_iter=20)
randomCV.fit(X_train, y_train)

best_n_estim      = randomCV.best_params_['n_estimators']
best_max_features = randomCV.best_params_['max_features']

rf_2 = RandomForestClassifier(n_estimators=best_n_estim,
                            max_features=best_max_features)

rf_2.fit(X_train, y_train)

with open(r'./Analisis_calidad_vino_blanco_ML/models/modelo.pkl', 'wb') as archivo:
    pickle.dump(rf_2, archivo)

with open(r'./Analisis_calidad_vino_blanco_ML/models/model_config.yaml', 'w') as c:
    yaml.dump(rf_2, c)