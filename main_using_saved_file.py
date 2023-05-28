# Multilayer sequencial
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from keras.saving.legacy.model_config import model_from_json
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV

forecasters = pd.read_csv("breast_entries.csv")
outputs = pd.read_csv("breast_output.csv")

scaler = MinMaxScaler()
forecasters_normalized = scaler.fit_transform(forecasters)

# PÒS TUNNING
# units = 8, activation = relu,

file = open('classifier_breast.json')
network_structure = file.read()
file.close()

classifier = model_from_json(network_structure)
classifier.load_weights('classifier_breast.h5')


predict = classifier.predict([forecasters.iloc[65].values.tolist()])

predict = (predict > 0.75)

print(predict)