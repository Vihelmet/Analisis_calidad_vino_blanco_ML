import pandas as pd
import numpy as np

# Cargar raw

vino = pd.read_csv(r'.\Analisis_calidad_vino_blanco_ML\data\raw\winequality-white.csv', delimiter= ';')

# Procesar features

# Guardar processed

vino.to_csv(r'.\Analisis_calidad_vino_blanco_ML\data\processed.csv')
vino.to_csv(r'.\Analisis_calidad_vino_blanco_ML\data\train.csv')
vino.to_csv(r'.\Analisis_calidad_vino_blanco_ML\data\test.csv')



