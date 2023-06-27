import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import pickle
import streamlit as st

with open(r'./Analisis_calidad_vino_blanco_ML/models/modelo_rf_def.pkl', 'rb') as modrf:
    randomfo = pickle.load(modrf)

with open(r'./Analisis_calidad_vino_blanco_ML/models/modelo_reglog.pkl', 'rb') as modrl:
    lgo = pickle.load(modrl)

with open(r'./Analisis_calidad_vino_blanco_ML/models/modelo_gb.pkl', 'rb') as modgb:
    gbo = pickle.load(modgb)

with open(r'./Analisis_calidad_vino_blanco_ML/models/modelo_svc.pkl', 'rb') as modsvc:
    svco = pickle.load(modsvc)

with open(r'./Analisis_calidad_vino_blanco_ML/models/modelo_knn.pkl', 'rb') as modknn:
    knno = pickle.load(modknn)


def classify(num):

    if num == 0:
        return 'Mala calidad'
    if num == 1:
        return 'Buena calidad'
    
def main():
    st.title('Entrenamiento modelos de predicción de la calidad del vino blanco')

    st.sidebar.header('Parámetros de entrada')

    def user_input_parameters(): #Poner todas las columnas
        fixedac = st.sidebar.slider('Fixed acidity', 3, 14)
        volaci = st.sidebar.slider('Volatile acidity', 0, 2)
        citricac = st.sidebar.slider('Citric acid', 0, 2)
        ressug = st.sidebar.slider('Residual sugar', 0, 66)
        chlorides = st.sidebar.slider('Chlorides', 0, 1)
        freesd = st.sidebar.slider('Free sulphur dioxide', 2, 289)
        totalsd = st.sidebar.slider('Total sulphur dioxide', 9, 440)
        density = st.sidebar.slider('Density', 0, 2)
        pH = st.sidebar.slider('pH', 2, 4)
        sulp = st.sidebar.slider('Sulphates', 0, 2)
        alcohol = st.sidebar.slider('Alcohol', 8, 15)
                    
        data = {'Fixed acidity': fixedac,
                'Volatile acidity': volaci,
                'Citric acid': citricac,
                'Residual sugar': ressug,
                'Chlorides': chlorides,
                'Free sulphur dioxide': freesd,
                'Total sulphur dioxide': totalsd,
                'Density': density,
                'pH': pH,
                'Sulphates': sulp,
                'Alcohol': alcohol,
                }
        
        features = pd.DataFrame(data, index=[0])
        return features
    
    df = user_input_parameters()

    option = ['Logistic Regression', 'Random Forest Classifier', 'Gradient Boost', 'SVM', 'KNN']
    model = st.sidebar.selectbox('¿Qué modelo deseas utilizar?', option)

    st.subheader('User Input Parameters')
    st.subheader(model)
    st.write(df)

    if st.button('RUN'):
        if model == 'Logistic Regression':
            st.success(classify(lgo.predict(df)))
        elif model == 'Random Forest Classifier':
            st.success(classify(randomfo.predict(df)))
        elif model == 'Gradient Boost':
            st.success(classify(gbo.predict(df)))
        elif model == 'SVM':
            st.success(classify(svco.predict(df)))
        elif model == 'KNN':
            st.success(classify(knno.predict(df)))


if __name__ == '__main__':
    main()
