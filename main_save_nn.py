# Multilayer sequencial
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
import matplotlib.pyplot as plt
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

classifier = Sequential()
classifier.add(Dense(units=8,
                         activation='relu',
                         kernel_initializer='normal',
                         input_dim=30))
# Vai zerar 20% da camada da segunda camada oculta
classifier.add(Dropout(0.2))
classifier.add(Dense(units=8,
                         activation='relu',
                         kernel_initializer='normal'))
# Vai zerar 20% da camada da segunda camada oculta
classifier.add(Dropout(0.2))
classifier.add(Dense(units=1, activation='sigmoid'))


classifier.compile(optimizer='adam', loss='binary_crossentropy',
                       metrics=['binary_accuracy'])

classifier.fit(forecasters, outputs, batch_size=10, epochs=100)

classifier_json = classifier.to_json()

with open('classifier_breast.json', 'w') as json_file:
    json_file.write(classifier_json)

classifier.save_weights('classifier_breast.h5')