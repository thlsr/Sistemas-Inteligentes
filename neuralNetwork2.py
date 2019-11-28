from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
import numpy as np

# # Carrega os dados que serao treinados na rede
# train = np.loadtxt("/content/drive/My Drive/Colab Notebooks/base.csv", delimiter=",")
# trainFeatures = train[:, 0:9]
# trainCrisis = train[:,9]

# # Carrega os dados que servirao para testar a acertividade da rede
# test = np.loadtxt("/content/drive/My Drive/Colab Notebooks/resultados.csv", delimiter=",")
# testFeatures = test[:, 0:9]
# testCrisis = test[:,9]

# Cria os dados de treino e teste de forma aleatoria
full = np.loadtxt("/content/drive/My Drive/Colab Notebooks/fullDataset.csv", delimiter=",")
train, test = train_test_split(full, test_size=0.3)

# Separa os dados de treino e teste nas features e resposta correspondente (crisis)
# Train
trainFeatures = train[:, 0:9]
trainCrisis = train[:,9]
# Test
testFeatures = test[:, 0:9]
testCrisis = test[:,9]

# REDE NEURAL

# Cria a arquitetura da rede neural
model = Sequential()
model.add(Dense(12, input_dim=9, activation='relu'))
model.add(Dense(15, activation='relu'))
# model.add(Dense(8, activation='relu'))
# model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Configura o modelo de treino
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy'])

# Treina o modelo para um numero fixo de epochs
model.fit(trainFeatures, trainCrisis, epochs=50, batch_size=10, validation_data=(testFeatures, testCrisis))
# model.fit(trainFeatures, trainCrisis, epochs=10, batch_size=10)
score = model.evaluate(trainFeatures, trainCrisis)
print("\nTrain %s: %.2f%%" % (model.metrics_names[1], score[1]*100))

test_loss, test_acc = model.evaluate(testFeatures, testCrisis)
print("Test accuracy: %f%%" % (test_acc*100))