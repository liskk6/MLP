import pandas as pd
import copy
import random

random.seed(15)
import math
import os


# Funkcja do pobrania wartosci
def getValues(data: pd.DataFrame, rowIndex: int, rowLen: int):
    res = []
    for i in range(rowLen):
        res.append(data.iloc[rowIndex, i])
    return res


# funkcja do oblicznie ile w ogole ma byc wag w mlp
def calculateAmoutOfWeights(layers: list):
    sum = 0
    for i in range(1, len(layers)):
        sum += layers[i]
        sum += layers[i - 1] * layers[i]
    return sum


# Funkcja do generowania wag
# Przyjmuje liste z warstwami i uwzglednia bias
# generuje liste z wagami dlugoscia [len(layers) - 1], gdzie dla kazdej warstwy tworzona jest osobna lista z wagami dla kazdego neurony
# np. jesli bylo podane [... , 3 ,4], to otrzymamy wagi pomiedzy warstwa ukryta a warstwa wyjsciowa jako [4 listy po 3 + 1 (jesli jest bias) elementy]
# warstwa -> neuron -> wagi
# Wypelnia 0, jesli ukazany jest momentum
def generateWeights(layers: list, isBias: bool, init: str):
    if isBias:
        bias = 1
    else:
        bias = 0

    result = []

    for i in range(1, len(layers)):
        weightsForLayer = []

        for j in range(layers[i]):
            weightsForNeuron = []

            for k in range(layers[i - 1] + bias):

                if init == "weights":
                    limit = 1 / math.sqrt(layers[i - 1])
                    weightsForNeuron.append(random.uniform(-limit, limit))
                elif init == "momentum":
                    weightsForNeuron.append(0.0)

            weightsForLayer.append(weightsForNeuron)
        result.append(weightsForLayer)

    return result


# Jesli random wlaczony losuje maxEra elementow sposrod wszystkich indeksow
# Jesli randow wylaczony dzile maxEra na 3,zeby trening byl prowadzony na takiej samej liczbie elementow kazdego typu
def generateIndexToTrain(isRandom: bool, dataLen: int, maxEra: int):
    trainIndices = []
    if isRandom:
        allIndices = [i for i in range(0, dataLen)]
        trainIndices = random.sample(allIndices, maxEra)
    else:
        amountOfEveryClass = int(maxEra / 3)
        for i in range(amountOfEveryClass):
            trainIndices.append(i)
            trainIndices.append(i + 50)
            trainIndices.append(i + 100)

    return trainIndices


# Funkcja aktywacji
def sigmoid(sum: float):
    if sum < -700:
        return 0

    if sum > 700:
        return 1

    return 1.0 / (1.0 + math.exp(-sum))


def sigmoidDerivative(sum: float):
    return sum * (1 - sum)


def forwardPropagation(weights, layers, isBias, originalValues):
    # Kopia ktora potem bedzie sie zmieniala
    values = originalValues.copy()

    # Wszystkie wartosci funkcji aktywacji
    sigmoidValuesForEveryLayer = []

    # Wszystkie dane podawane w przod
    inputsForEveryLayer = []

    # Jesli bias jest, to dodajemy 1 no koncu zeby potem po prostu
    if isBias:
        values.append(1.0)

    for layer in range(len(layers) - 1):
        # Dodajemy kopie pobranych wartosci
        inputsForEveryLayer.append(values.copy())

        # Lista przechowujaca wartosci ten warstwy po funkcji aktywacji
        layer_values = []

        for neuron in range(len(weights[layer])):
            neuronSum = 0
            # Lista z wagami wzgledem neuronu do wszystkich nastepnych neuronow
            neuronWeights = weights[layer][neuron]
            for valueIndex in range(len(values)):
                neuronSum += values[valueIndex] * neuronWeights[valueIndex]

            layer_values.append(sigmoid(neuronSum))

        # Kopjujemy wyliczone wartosci
        values = layer_values.copy()
        sigmoidValuesForEveryLayer.append(layer_values.copy())

        if isBias:
            values.append(1.0)
    # Po obliczeniu wartosci usuwamy bias
    if isBias:
        values = values[:-1]

    return values, sigmoidValuesForEveryLayer, inputsForEveryLayer


def generateStatsForSave(confusionMatrix: list, mode:str):
    correctSum = 0
    for i in range(len(confusionMatrix)):
        correctSum += confusionMatrix[i][i]

    precision = [0.0 for _ in range(len(confusionMatrix))]
    recall = [0.0 for _ in range(len(confusionMatrix))]
    fmeasure = [0.0 for _ in range(len(confusionMatrix))]
    if mode == "Iris":
        classesList = ["Setosa", "Versicolor", "Virginica"]
    else:
        classesList = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    for rowInd in range(len(confusionMatrix)):
        TP = confusionMatrix[rowInd][rowInd]
        FP = 0.0
        FN = 0.0
        for colInd in range(len(confusionMatrix)):
            if colInd == rowInd:
                continue
            else:
                FP += confusionMatrix[colInd][rowInd]
                FN += confusionMatrix[rowInd][colInd]

        if (TP + FP == 0):
            precisionValue = 0
        else:
            precisionValue = TP / (TP + FP)

        if (TP + FN == 0):
            recallValue = 0
        else:
            recallValue = TP / (TP + FN)
        precision[rowInd] = precisionValue
        recall[rowInd] = recallValue

        if (precisionValue + recallValue == 0):
            fmeasure[rowInd] = 0
        else:
            fmeasure[rowInd] = (2 * ((precisionValue * recallValue) / (precisionValue + recallValue)))

    return classesList, fmeasure, precision, recall









