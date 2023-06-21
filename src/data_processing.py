import pandas as pd
import numpy as np
import seaborn as sns

vino = pd.read_csv('../data/raw/winequality-white.csv', delimiter= ';')
vino

vino_features = vino.drop(columns='quality')
vino_features

vino_target = vino['quality']