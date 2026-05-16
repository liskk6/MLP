import pandas as pd
from datetime import datetime
import random
import os
import numpy as np
from mlp_functions import getValues, generateWeights, sigmoid, sigmoidDerivative, forwardPropagation,saveMLP,generateStatsForSave


def MLP(data: pd.DataFrame, layers: list, maxEpochs: int, isRandom: bool, isBias: bool, learningSpeed: float,
        momentumCoef: float, isSave :bool,ifPrint: bool, logsFileName:str,saveFileName:str, weightsParam=None):
    # Sprawdza czy dane wprowadzono poprawnie
    if (maxEpochs <= 0) or (momentumCoef < 0.0) or (learningSpeed < 0.0) or len(layers) < 2:
        return None
    # Slownik z poprawnymi odpowiedziami
    classes = {"Iris-setosa": [1, 0, 0], "Iris-versicolor": [0, 1, 0], "Iris-virginica": [0, 0, 1]}
    path = f'../stats/{logsFileName}.txt'
    # Wagi generowane przy uzyciu funkcji lub przyjmowane z funkcji
    if weightsParam is not None:
        weights = weightsParam
    else:
        weights = generateWeights(layers, isBias, "weights")

    # Lista zer dla momentum
    momentum = generateWeights(layers, isBias, "momentum")

    # Lista indeksow od 0 do 150)
    trainIndices = [i for i in range(0, len(data))]

    # Warunek uzycia momentum. Jesli wspolczynnik < 0 momentum nie bedzie uwzgledniony
    useMomentum = momentumCoef > 0.0

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as file:
        file.write(
            f"MLP settings.\nLayers: {layers}; Epochs: {maxEpochs}; Random: {isRandom}; Bias: {isBias}; Learning speed: {learningSpeed}; Momentum coefficient: {momentumCoef}\n\n")

    for epoch in range(1, maxEpochs+1):
        # Macierz pomylek
        confusionMatrix = [[0 for _ in range(layers[-1])] for _ in range(layers[-1])]

        # Jesli wlaczona losowosc
        if isRandom:
            trainIndices = random.sample(trainIndices, len(data))

        for era in trainIndices:
            # Pobrane dane
            originalValues = getValues(data, era, layers[0])

            values, sigmoidValuesForEveryLayer, inputsForEveryLayer = forwardPropagation(weights, layers, isBias,
                                                                                         originalValues)

            # Pobiweramy poprawna odpowiedz
            irisType = classes[data.iloc[era, 4]]

            # Lista ze wszytkimi deltami
            allDeltasLayer = []

            # Lista delt dla warstwy wyjsciowejew
            outPutLayer = []

            for value in range(layers[-1]):
                # Delta dla warstwy wyjsciowej = Poprawna opd - faktyczna odp * pochodna funkci aktywaacji(faktyczna odp)
                outPutLayer.append((irisType[value] - values[value]) * sigmoidDerivative(values[value]))
            allDeltasLayer.append(outPutLayer)

            # Oblicza delte tylko warstw ukrytych
            for layerIdx in range(len(layers) - 2, 0, -1):
                deltasLayer = []
                for currNeuron in range((layers[layerIdx])):

                    deltasSum = 0
                    for nextNeuron in range(len(allDeltasLayer[-1])):
                        # Wagi razy wartosc delty
                        deltasSum += weights[layerIdx][nextNeuron][currNeuron] * allDeltasLayer[-1][nextNeuron]

                    # Wylicza pochodna
                    derivative = sigmoidDerivative(sigmoidValuesForEveryLayer[layerIdx - 1][currNeuron])
                    # Delta
                    delta = deltasSum * derivative
                    deltasLayer.append(delta)

                allDeltasLayer.append(deltasLayer)

            # Odwraca liste
            allDeltasLayer = allDeltasLayer[::-1]

            # Pobieramy indeks poprawnej odpowiedzi
            correctAnswer = irisType.index(max(irisType))
            # pobieramy indeks wyliczonej odpowiedzi
            mlpAnswer = values.index(max(values))

            # Dodajemy do macierzy pomylek
            confusionMatrix[correctAnswer][mlpAnswer] += 1

            # Oblicza nowe wagi i ewentualnie momentum
            for layerInd in range(len(layers) - 1):

                for currNeuron in range(len(inputsForEveryLayer[layerInd])):

                    for nextNeuron in range(len(allDeltasLayer[layerInd])):
                        # Wartosc gradientu
                        gradient = learningSpeed * allDeltasLayer[layerInd][nextNeuron] * inputsForEveryLayer[layerInd][
                            currNeuron]
                        # Uwzglednia momentum
                        if useMomentum:
                            step = gradient + (momentum[layerInd][nextNeuron][currNeuron] * momentumCoef)
                            weights[layerInd][nextNeuron][currNeuron] += step
                            momentum[layerInd][nextNeuron][currNeuron] = step
                        else:
                            weights[layerInd][nextNeuron][currNeuron] += gradient
        correctSum = 0
        for i in range(len(confusionMatrix)):
            correctSum += confusionMatrix[i][i]
        if ifPrint:
            if epoch % 100 == 0:
                print(
                    f"Epoch - {epoch}/{maxEpochs}. Matrix -{confusionMatrix} - "
                    f"{correctSum}/{len(data)} {round(correctSum / len(data) * 100, 3)}%")

        if (epoch % 20 == 0) or epoch == maxEpochs:
            classesList, fmeasure, precision, recall = generateStatsForSave(confusionMatrix)
            now = datetime.now()
            t = now.strftime("%H:%M:%S")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, 'a', encoding='utf-8') as file:
                file.write(f"\n{epoch}|{maxEpochs} - Epochs. Time: {t}\n")
                for irisType in range(len(classesList)):
                    formattedMatrixRow = "[" + ", ".join([f"{val:>2}" for val in confusionMatrix[irisType]]) + "]"
                    file.write(
                        f"{formattedMatrixRow} {classesList[irisType]:<10} --- Precision = {round(precision[irisType], 3):<7}Recall = {round(recall[irisType], 3):<6}FMeasure = {round(fmeasure[irisType], 3):<5}; True Positive = {confusionMatrix[irisType][irisType]}|{int(len(data) / len(confusionMatrix))};\n")
                file.write(
                    f"Confusion Matrix: {correctSum}|{len(data)} ({round(correctSum / len(data) * 100, 3)}%)\n\n")

    if isSave:
        path = f'../saves/{saveFileName}.txt'
        saveMLP(path, layers, maxEpochs, isRandom, isBias, learningSpeed, momentumCoef,  weights)


    return weights


def MLP_test(data: pd.DataFrame, rowIndex: int, layers: list, isBias: bool, weightsParam=None):
    classes = {"Iris-setosa": [1, 0, 0], "Iris-versicolor": [0, 1, 0], "Iris-virginica": [0, 0, 1]}
    confusionMatrix = [[0 for _ in range(layers[-1])] for _ in range(layers[-1])]

    originalValues = getValues(data, rowIndex, layers[0])

    values, sigmoidValuesForEveryLayer, inputsForEveryLayer = forwardPropagation(weightsParam, layers, isBias,
                                                                                 originalValues)

    irisType = classes[data.iloc[rowIndex, 4]]

    # Pobieramy indeks poprawnej odpowiedzi
    correctAnswer = irisType.index(max(irisType))
    # pobieramy indeks wyliczonej odpowiedzi
    mlpAnswer = values.index(max(values))

    # Dodajemy do macierzy pomylek
    confusionMatrix[correctAnswer][mlpAnswer] += 1

    errorMade = []
    MSE = 0
    for i in range(len(values)):
        error = irisType[i] - values[i]
        MSE += error * error
        errorMade.append(error)

    path = '../stats/statsTesting.txt'
    roundedWeights = [[[round(float(w), 5) for w in neuron] for neuron in layer] for layer in weightsParam]
    now = datetime.now()
    t = now.strftime("%H:%M:%S")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as file:
        file.write(
            f"MLP settings:\nMode: Testing; Time: {t} \nLayers: {layers}; Bias: {isBias};\nWeights: {roundedWeights} ;  \n\n")
        file.write(f"Original values: {[round(float(i), 5) for i in originalValues]};\n")
        file.write(f"Output values : {[round(float(i), 5) for i in values]};  Response pattern: {irisType}; \n")
        file.write(f"MSE: {round(MSE, 4):<4};\n")
        for i in range(len(errorMade)):
            file.write(f"Neuron: {i + 1} - Error: {errorMade[i]}\n")

        file.write(f"Output layer weights:{roundedWeights[-1]}\n")

        roundedWeightsRev = roundedWeights[::-1]
        for i in range(len(weightsParam) - 1, 0, -1):
            file.write(f"Hidden neurons layer weights: {roundedWeightsRev[i]}\n")
        sigmoidValuesForEveryLayer = [[round(float(w), 5) for w in neuron] for neuron in sigmoidValuesForEveryLayer]
        sigmoidValuesForEveryLayerRev = sigmoidValuesForEveryLayer[::-1]
        for i in range(len(inputsForEveryLayer) - 1, 0, -1):
            file.write(f"Hidden neurons values: {sigmoidValuesForEveryLayerRev[i]}\n")

    return errorMade
