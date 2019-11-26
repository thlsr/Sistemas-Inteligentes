# Colunas:
# 1.	Numero especifico do pais
# 2.	Codigo de 3 letras do pais
# 3.	Nome do pais
# 4.	Ano
# 5. 	Se ocorreu crise sistemica ou nao (1 ou 0)
# 6. 	Comparativo do valor da moeda do pais com o dolar
# 7.	Divida publica domestica (1 ou 0)
# 8.	Divida publica externa (1 ou 0)
# 9. 	Divida total em comparacao com o PIB
# 10.	Inflacao anual
# 11.	Independente (1 ou 0)
# 12.	Crise monetaria (1 ou 0)
# 13.	Crise inflacionaria (1 ou 0)
# 14. 	Crise bancaria (1 ou 0)



import csv
from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure import SigmoidLayer


with open('base.csv', 'rt') as csvbase:
    base = csv.reader(csvbase)
    linha = 0
    matrixDados = [[0] * 9 for i in range(1007)]
    matrixCriseBanc = [[0] for i in range(1007)]
    for row1 in base:
        for i in range(9):
            matrixDados[linha][i] = float(row1[i])
        matrixCriseBanc[linha] = float(row1[9])

        linha = linha +1

    csvbase.close()

with open('resultados.csv', 'rt') as csvteste:
    teste = csv.reader(csvteste)
    l = 0
    matrixTesteInput = [[0] * 9 for i in range(52)]
    matrixTesteOutput = [[0] for i in range(52)]
    for row2 in teste:
        for i in range(9):
            matrixTesteInput[l][i] = float(row2[i])
        matrixTesteOutput[l] = float(row2[9])

        l = l+1
    csvteste.close()


ds = SupervisedDataSet(9, 1)

# acrescenta os dados de treino
for j in range(1007):
    ds.addSample((matrixDados[j][0], matrixDados[j][1], matrixDados[j][2], matrixDados[j][3], matrixDados[j][4],
                  matrixDados[j][5], matrixDados[j][6], matrixDados[j][7], matrixDados[j][8]),
                 matrixCriseBanc[j])

# cria a arquitetura da rede neural
nn = buildNetwork(9, 20, 20, 1, hiddenclass=SigmoidLayer, bias=True)

# nn = buildNetwork(9, 1)

trainer = BackpropTrainer(nn, ds)

import time
inicio = time.time()
######## aqui se define o numero de treinos que serao executados ########################
for i in range(100):
    erro = trainer.train()
    print erro
    # print trainer.train()
fim = time.time()
print ('O treino demorou:', fim-inicio)
########################################################
matrixActivations = [[0] for i in range(52)]

for i in range(52):
    matrixActivations[i] = nn.activate([matrixTesteInput[i][0], matrixTesteInput[i][1], matrixTesteInput[i][2],
                                        matrixTesteInput[i][3], matrixTesteInput[i][4], matrixTesteInput[i][5],
                                        matrixTesteInput[i][6], matrixTesteInput[i][7], matrixTesteInput[i][8]])

acertos = 0
for i in range(52):
    # print 'Previsao de crise bancaria na linha', i, 'foi:', matrixActivations[i],'||| Sendo que o resultado esperado era:', matrixTesteOutput[i]
    diferenca = abs(matrixTesteOutput[i] - matrixActivations[i])
    print diferenca, '||', erro
    if diferenca < erro:
        acertos += 1
print 'O numero de acertos foi de:', (float(acertos)/52)*100,'%'

