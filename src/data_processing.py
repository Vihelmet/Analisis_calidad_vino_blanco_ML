import pandas as pd
import numpy as np

vino = pd.read_csv(r'.\Analisis_calidad_vino_blanco_ML\data\raw\winequality-white.csv', delimiter= ';')

vino['good quality'] = [1 if x > 5 else 0 for x in vino.quality]

vino.to_csv(r'.\Analisis_calidad_vino_blanco_ML\data\processed.csv')
vino.to_csv(r'.\Analisis_calidad_vino_blanco_ML\data\train.csv')
vino.to_csv(r'.\Analisis_calidad_vino_blanco_ML\data\test.csv')



