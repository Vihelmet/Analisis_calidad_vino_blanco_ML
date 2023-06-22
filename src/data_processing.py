from variables import *
import pandas as pd
import numpy as np
import seaborn as sns

vino = pd.read_csv('../data/raw/winequality-white.csv', delimiter= ';')
vino

vino_features = vino.drop(columns='quality')
vino_features

vino_target = vino['quality']

def csv(nombre_archivo, archivo_guardar):


    ruta_archivo = '../data/' + nombre_archivo + '.csv'
    archivo_guardar.to_csv(ruta_archivo, index=False)

#csv("test", vino_features)
#csv("train", vino)