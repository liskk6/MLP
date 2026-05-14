import pandas as pd
import random
import math

#Funkcja do pobrania wartosci
def getValues(data: pd.DataFrame,rowIndex: int, rowLen:int ):
    res = []
    for i in range(rowLen):
        res.append(data.iloc[rowIndex, i])
    return res



#funkcja do oblicznie ile w ogole ma byc wag w mlp
def calculateAmoutOfWeights(layers: list):
    sum = 0
    for i in range(1, len(layers)):
        sum += layers[i]
        sum += layers[i-1] * layers[i]
    return sum



# Funkcja do generowania wag
# Przyjmuje liste z warstwami i uwzglednia bias
# generuje liste z wagami dlugoscia [len(layers) - 1], gdzie dla kazdej warstwy tworzona jest osobna lista z wagami dla kazdego neurony
# np. jesli bylo podane [... , 3 ,4], to otrzymamy wagi pomiedzy warstwa ukryta a warstwa wyjsciowa jako [4 listy po 3 + 1 (jesli jest bias) elementy]
# warstwa -> neuron -> wagi
def generateWeights(layers: list, isBias : bool):

    if isBias:
        bias = 1
    else:
        bias = 0

    result = []
    for i in range(1, len(layers)):
        weightsForLayer = []

        for j in range(layers[i]):
            weightsForNeuron = []

            for k in range(layers[i-1] + bias):
               weightsForNeuron.append(random.uniform(-1.0, 1.0))

            weightsForLayer.append(weightsForNeuron)
        result.append(weightsForLayer)

    return result


#Jesli random wlaczony losuje maxEra elementow sposrod wszystkich indeksow
#Jesli randow wylaczony dzile maxEra na 3,zeby trening byl prowadzony na takiej samej liczbie elementow kazdego typu
def generateIndexToTrain(isRandom: bool, dataLen: int, maxEra: int):
    trainIndices = []
    if isRandom:
        allIndices = [i for i in range(0, dataLen)]
        trainIndices = random.sample(allIndices,maxEra)
    else:
        amountOfEveryClass = int(maxEra / 3)
        for i in range(amountOfEveryClass):
            trainIndices.append(i)
            trainIndices.append(i + 50)
            trainIndices.append(i + 100)

    return trainIndices

#Funkcja aktywacji
def sigmoid(sum: float):
    if sum < -500:
        return 0

    return 1.0 / (1.0 + math.exp(-sum))

def sigmoidDerivative(sum:float):
    return sigmoid(sum) * (1 - sigmoid(sum))