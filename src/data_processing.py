from variables import *
import pandas as pd
import numpy as np
import seaborn as sns
import os


current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
print(parent_dir)


vino = pd.read_csv(parent_dir + '/data/raw/winequality-white.csv', delimiter= ';')
vino

vino_features = vino.drop(columns=['fixed acidity', 'citric acid', 'residual sugar', 'free sulfur dioxide', 'total sulfur dioxide', 'pH', 'sulphates','quality'])
vino_features

vino_target = vino['quality']

# def csv(nombre_archivo, archivo_guardar):


#     ruta_archivo = './data/' + nombre_archivo + '.csv'
#     archivo_guardar.to_csv(ruta_archivo, index=False)

# csv("test", vino)
# csv("train", vino)