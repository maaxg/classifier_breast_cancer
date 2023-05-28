# Multilayer sequencial
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

forecasters = pd.read_csv("breast_entries.csv")
outputs = pd.read_csv("breast_output.csv")

# Test size - Significa que vamos utulizar 75% dos dados para treino e 25% para teste
forecasters_training, \
    forecasters_test, \
    outputs_training, \
    outputs_test = \
    train_test_split(forecasters, outputs, test_size=0.25)

classifier = Sequential()
# Neurônios totalmente conectados de forma sequencial.
# Unit = (30+1) /2 - math.ceil(forecasters_columns_length + outputs_columns_length / 2)

# Activation - função de ativação - relu - rectifier linear unit
# - por experimentação para o meu projeto ele deu um resultado melhor que a sigmoide ( Não testei com a hiperbolica )

# input_dim = 30 neurônios para a primeira camada oculta

# kernel_initializer = distribui de forma uniforme
# Camada de entrada - Oculta
classifier.add(Dense(units = 16,
                     activation= 'relu',
                     kernel_initializer= 'random_uniform',
                     input_dim = 30))
# Oculta
classifier.add(Dense(units = 16,
                     activation= 'relu',
                     kernel_initializer= 'random_uniform'))

# Por ser binário só precisamento de 1 neurônio na camada de saída
#activiation o sigmoid sempre retorna 0 ou 1, ele normaliza a saída para algo próximo disso
# o que torna as coisas mais fáceis visto que temos uma saída binária
classifier.add(Dense(units = 1, activation = 'sigmoid'))

# Adam é uma otimização da descida do gradiente estocástico

# learning rate => taxa de aprendizagem
# decay => valor de decaimento da taxa de aprendizagem
optimizer = Adam(lr = 0.001,    decay = 0.0001, clipvalue = 0.5)


classifier.compile(optimizer = optimizer, loss= 'binary_crossentropy',
                   metrics = ['binary_accuracy'])

# Batch size vai verificar o erro para 10 registros e ajustar os pesos
history = classifier.fit(forecasters_training, outputs_training, batch_size = 10, epochs = 200)

# Visualização dos pesos da primeira camada
weights0 = classifier.layers[0].get_weights()

predictions = classifier.predict(forecasters_test)

predictions = (predictions > 0.5)

accuracy = accuracy_score(outputs_test, predictions)

matrix = confusion_matrix(outputs_test, predictions)

result = classifier.evaluate(forecasters_test, outputs_test)

# Access the training history
training_loss = history.history['loss']
training_accuracy = history.history['binary_accuracy']

# Plotting the behavior of the network during training
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(range(1, len(training_loss) + 1), training_loss)
plt.title('Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.subplot(1, 2, 2)
plt.plot(range(1, len(training_accuracy) + 1), training_accuracy)
plt.title('Training Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.tight_layout()

# Conffusion matrix
plt.figure(figsize=(6, 4))
sns.heatmap(matrix, annot=True, cmap='Blues', fmt='d')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')

plt.show()