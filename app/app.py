import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import pickle

with open(r'./Analisis_calidad_vino_blanco_ML/models/modelo_rf_def.pkl', 'rb') as modelorf:
    randomf = pickle.load(modelorf)

with open(r'./Analisis_calidad_vino_blanco_ML/models/modelo_rf_def.pkl', 'rb') as modelorf:
    randomf = pickle.load(modelorf)