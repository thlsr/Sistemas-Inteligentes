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
import pybrain
from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer


with open('teste1.csv', 'rt') as csvfile:
    data = csv.reader(csvfile)
    linha = 0
    matrixDados = [[0] * 10 for i in range(1007)]
    matrixCriseBanc = [[0] for i in range(1007)]
    for row in data:
        #print row[9]
        for i in range(9):
            matrixDados[linha][i] = float(row[i])
        matrixCriseBanc[linha] = float(row[9])

        linha = linha +1

csvfile.close()

ds = SupervisedDataSet(9, 1)

# print(matrixDados)
# print(matrixCriseBanc)

#acrescenta os dados de treino
for j in range(1007):
    ds.addSample((matrixDados[j][0], matrixDados[j][1], matrixDados[j][2], matrixDados[j][3], matrixDados[j][4],
                  matrixDados[j][5], matrixDados[j][6], matrixDados[j][7], matrixDados[j][8]),
                 matrixCriseBanc[j])

#nn = buildNetwork(9, 3, 1, bias=True)
nn = buildNetwork(9, 1)

trainer = BackpropTrainer(nn, ds)

for i in range(20):
    print trainer.train()


a = nn.activate([0, 0.1021841633, 0, 0, 0, 0.000000205641777, 1, 0, 0])
b = nn.activate([0, 0.1049333009, 0, 0, 0, 0.0000004054626456, 1, 0, 0])
c = nn.activate([0, 0.1049953739, 0, 0, 0, 0.0000001480238797, 1, 0, 0])
d = nn.activate([0, 0.1181915528, 0, 0, 0, 0.0000001326530437, 1, 0, 0])
e = nn.activate([0, 0.128001094, 0, 0, 0, 0.0000006131963115, 1, 0, 0])
f = nn.activate([0, 0.1287452571, 0, 0, 0, 0.0000004677190792, 1, 0, 0])
g = nn.activate([0, 0.1310750979, 0, 0, 0, 0.0000003993688822, 1, 0, 0])
h = nn.activate([0, 0.1363551833, 0, 0, 0, 0.0000003317917746, 1, 0, 0])
i = nn.activate([0, 0.6811190973, 0, 1, 0, 0.00000005434363633, 1, 0, 0])
j = nn.activate([0, 0.667955131, 0, 1, 0, 0.0000002671251212, 1, 0, 0])
k = nn.activate([0, 0.6390399535, 0, 1, 0, 0.000000297957745, 1, 0, 0])
l = nn.activate([0, 0.7258874886, 0, 1, 0, 0.0000005267922035, 1, 0, 0])
m = nn.activate([0, 0, 0, 1, 0, 0.0000002022765643, 1, 0, 0])
n = nn.activate([0, 0, 0, 1, 0, 1.3, 1, 0, 0])
o = nn.activate([0, 0, 0, 0, 0, 0.0000001175095868, 1, 0, 0])

print('Previsao de crise bancaria:', a,'Resultado esperado:0')
print('Previsao de crise bancaria:', b,'Resultado esperado:0')
print('Previsao de crise bancaria:', c,'Resultado esperado:0')
print('Previsao de crise bancaria:', d,'Resultado esperado:0')
print('Previsao de crise bancaria:', e,'Resultado esperado:0')
print('Previsao de crise bancaria:', f,'Resultado esperado:0')
print('Previsao de crise bancaria:', g,'Resultado esperado:0')
print('Previsao de crise bancaria:', h,'Resultado esperado:0')
print('Previsao de crise bancaria:', i,'Resultado esperado:0')
print('Previsao de crise bancaria:', j,'Resultado esperado:0')
print('Previsao de crise bancaria:', k,'Resultado esperado:0')
print('Previsao de crise bancaria:', l,'Resultado esperado:0')
print('Previsao de crise bancaria:', m,'Resultado esperado:0')
print('Previsao de crise bancaria:', n,'Resultado esperado:0')
print('Previsao de crise bancaria:', o,'Resultado esperado:0')
