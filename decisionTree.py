import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier

from sklearn.tree import export_graphviz
from pydotplus.graphviz import graph_from_dot_data

from sklearn import tree
import graphviz

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


# DECISION TREE CLASSIFIER

# Inicializa a arvore
tree = DecisionTreeClassifier(criterion='entropy', random_state=1)

# Treina os dados na arvore
tree.fit(trainFeatures, trainCrisis)

# Utiliza os dados de teste para verificar a acuracia da arvore
predict = tree.predict(testFeatures)
# print(predict)

acertos = 0
for i in range(52):
  if(testCrisis[i] == predict[i]):
    acertos +=1

print("Acertos: %f%%" % (float(acertos/52)*100))

# Serve apenas para exportar a estrutura da arvore em formato de imagem
dot_data = export_graphviz(
    tree, filled=True, rounded=True,
    class_names=['naoCrise', 'Crise'],
    feature_names=['criseSist','cambioDolar','dividaPublInt','dividaPubExt','dividaPib','inflAnual','indep','criseMone','criseInfl'],
    out_file=None    
)
graph = graph_from_dot_data(dot_data)
graph.write_png('classifier.png')


# DECISION TREE REGRESSOR
# regressor = DecisionTreeRegressor(random_state=0)
# regressor.fit(trainFeatures, trainCrisis)

# pred = regressor.predict(testFeatures)
# print(pred)

# acertos = 0
# for i in range(52):
#   if(testCrisis[i] == pred[i]):
#     acertos +=1

# print("Acertos: %f" % (float(acertos/52)*100))

# dot_data = export_graphviz(regressor, out_file='tree.dot', 
#                 feature_names=['criseSist','cambioDolar','dividaPublInt','dividaPubExt','dividaPib',
#                                'inflAnual','indep','criseMone','criseInfl'])

# graph = graph_from_dot_data(dot_data)
# graph.write_png('regressor.png')