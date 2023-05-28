# Multilayer sequencial - Crossed validation
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

forecasters = pd.read_csv("breast_entries.csv")
outputs = pd.read_csv("breast_output.csv")

scaler = MinMaxScaler()
forecasters_normalized = scaler.fit_transform(forecasters)


def createNetwork():
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
    return classifier


classifier = KerasClassifier(build_fn=createNetwork, epochs=100, batch_size=10)

results = []
epochs = range(1, 101)
for epoch in epochs:
    scores = cross_val_score(estimator=classifier, X=forecasters_normalized, y=outputs, cv=10, scoring='accuracy')
    results.append(scores)

results = np.array(results)

#results_average = results.mean();
#standard_deviation = results.std();

results_average = np.mean(results, axis=1)
standard_deviation = np.std(results, axis=1)

classifier_json = classifier.to_json()

with open('classifier_breast_crossed.json', 'w') as json_file:
    json_file.write(classifier_json)

classifier.save_weights('classifier_breast_crossed.h5')

# Precision, Recall, Accuracy
plt.scatter(epochs, results_average, color='red', label='Individual Scores')
plt.errorbar(epochs, results_average, yerr=standard_deviation, fmt='o', capsize=3)
plt.title('Average Accuracy and Standard Deviation per Epoch')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()
