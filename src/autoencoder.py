from mlp_functions import generateWeights,getValues, forwardPropagation,sigmoidDerivative, generateStatsForSave
import random
import os
from datetime import datetime

def autoencoder(maxEpochs:int ,isRandom:bool, isBias:bool,learningSpeed: float, momentumCoef: float):
    if (maxEpochs <= 0) or (momentumCoef < 0.0) or (learningSpeed < 0.0) :
        return None
    layers = [4,2,4]
    data = [[1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,1]]
    weights = generateWeights(layers, isBias, 'weights')
    momentum = generateWeights(layers, isBias, "momentum")
    path = path = f'../stats/autoencoder.txt'
    trainIndices = [i for i in range(0, len(data))]

    useMomentum = momentumCoef > 0.0

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as file:
        file.write(
            f"MLP settings.\nLayers: {layers}; Epochs: {maxEpochs}; Random: {isRandom}; Bias: {isBias}; Learning speed: {learningSpeed}; Momentum coefficient: {momentumCoef}\n\n")

    for epoch in range(1, maxEpochs + 1):
        confusionMatrix = [[0 for _ in range(layers[-1])] for _ in range(layers[-1])]

        if isRandom:
            trainIndices = random.sample(trainIndices, len(data))

        for era in trainIndices:
            # Pobrane dane
            originalValues = data[era]

            values, sigmoidValuesForEveryLayer, inputsForEveryLayer = forwardPropagation(weights, layers, isBias,originalValues)

            allDeltasLayer = []

            # Lista delt dla warstwy wyjsciowejew
            outPutLayer = []

            for value in range(layers[-1]):
                # Delta dla warstwy wyjsciowej = Poprawna opd - faktyczna odp * pochodna funkci aktywaacji(faktyczna odp)
                outPutLayer.append((originalValues[value] - values[value]) * sigmoidDerivative(values[value]))
            allDeltasLayer.append(outPutLayer)

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

            allDeltasLayer = allDeltasLayer[::-1]

            # Pobieramy indeks poprawnej odpowiedzi
            correctAnswer = originalValues.index(max(originalValues))
            # pobieramy indeks wyliczonej odpowiedzi
            mlpAnswer = values.index(max(values))

            # Dodajemy do macierzy pomylek
            confusionMatrix[correctAnswer][mlpAnswer] += 1

            for layerInd in range(len(layers) - 1):

                for currNeuron in range(len(inputsForEveryLayer[layerInd])):

                    for nextNeuron in range(len(allDeltasLayer[layerInd])):
                        # Wartosc gradientu
                        gradient = learningSpeed * allDeltasLayer[layerInd][nextNeuron] * \
                                       inputsForEveryLayer[layerInd][
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
        if epoch % 50 == 0:
            print(
                 f"Epoch - {epoch}/{maxEpochs}. Matrix -{confusionMatrix} - "
                f"{correctSum}/{len(data)} {round(correctSum / len(data) * 100, 3)}%")

        if (epoch % 20 == 0) or epoch == maxEpochs or epoch == 0:
            classesList, fmeasure, precision, recall = generateStatsForSave(confusionMatrix, 'autoencoder')
            now = datetime.now()
            t = now.strftime("%H:%M:%S")

            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, 'a', encoding='utf-8') as file:
                file.write(f"\n{epoch}|{maxEpochs} - Epochs. Time: {t}\n")
                for irisType in range(len(data)):
                    formattedMatrixRow = "[" + ", ".join([f"{val:>2}" for val in confusionMatrix[irisType]]) + "]"
                    file.write(
                        f"{formattedMatrixRow} {classesList[irisType]} --- Precision = {round(precision[irisType], 3):<7}Recall = {round(recall[irisType], 3):<6}FMeasure = {round(fmeasure[irisType], 3):<5}; True Positive = {confusionMatrix[irisType][irisType]}|{int(len(data) / len(confusionMatrix))};\n")
                file.write(
                    f"Confusion Matrix: {correctSum}|{len(data)} ({round(correctSum / len(data) * 100, 3)}%)\n\n")

    return weights