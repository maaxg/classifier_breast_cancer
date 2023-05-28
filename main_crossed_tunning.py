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


def createNetwork(optimizer, loss, kernel_initializer, activation, neurons):
    classifier = Sequential()
    classifier.add(Dense(units=neurons,
                         activation=activation,
                         kernel_initializer=kernel_initializer,
                         input_dim=30))
    # Vai zerar 20% da camada da segunda camada oculta
    classifier.add(Dropout(0.2))
    classifier.add(Dense(units=neurons,
                         activation=activation,
                         kernel_initializer=kernel_initializer))
    # Vai zerar 20% da camada da segunda camada oculta
    classifier.add(Dropout(0.2))
    classifier.add(Dense(units=1, activation='sigmoid'))

    classifier.compile(optimizer=optimizer, loss=loss, metrics=['binary_accuracy'])
    return classifier

classifier = KerasClassifier(build_fn=createNetwork)

# descida do gradiente estocastica se baseia no batch_size o número de vezes q ele precisa passar por um registro até atualizar os pesos.
parameters = {'batch_size': [10, 20], 'epochs': [10, 20],
              'optimizer': ['adam'],
              'loss': ['binary_crossentropy'],
              'kernel_initializer': ['random_uniform', 'normal'],
              'activation': ['relu'],
              'neurons': [16, 8]}

grid_search = GridSearchCV(estimator=classifier, param_grid=parameters, scoring='accuracy', cv=5)

grid_search = grid_search.fit(forecasters, outputs)

best_params = grid_search.best_params_
best_score = grid_search.best_score_

