import pandas as pd
import copy
import random
import numpy as np
from mlp_functions import getValues,generateWeights,generateIndexToTrain, sigmoid, sigmoidDerivative


def MLP(data : pd.DataFrame, layers: list, maxEpochs: int, isRandom : bool, isBias: bool, learningSpeed: float, momentumCoef : float, weightsParam=None):

    #Sprawdza czy dane wprowadzono poprawnie
    if (maxEpochs < 0) or (maxEpochs < 0) or (learningSpeed < 0):
        return None
    classes = {"Iris-setosa": [1, 0, 0], "Iris-versicolor": [0, 1, 0], "Iris-virginica": [0, 0, 1]}

    #Wagi generowane przy uzyciu funkcji

    if weightsParam is not None:
        weights = weightsParam
    else:
        weights = generateWeights(layers, isBias, "weights")

    momentum = generateWeights(layers, isBias, "momentum")

    trainIndices =  [i for i in range(0, 150)]

    useMomentum = momentumCoef > 0.0



    for epoch in range(maxEpochs):

        if isRandom:
            trainIndices = random.sample(trainIndices,150)

        for era in trainIndices:
            #Pobiera dane
            originalValues = getValues(data, era, layers[0])
            values = copy.deepcopy(originalValues)

            sigmoidValuesForEveryLayer = []
            inputsForEveryLayer = []

            #Jesli bias jest, to dodajemy 1 no koncu zeby potem po prostu
            if isBias:
                values.append(1.0)

            for layer in range(len(layers) - 1):
                inputsForEveryLayer.append(values.copy())
                layer_values = []

                for neuron in range(len(weights[layer])):
                    neuronSum = 0
                    neuronWeights = weights[layer][neuron]
                    for valueIndex in range(len(values)):
                        neuronSum += values[valueIndex] * neuronWeights[valueIndex]

                    layer_values.append(sigmoid(neuronSum))

                values = layer_values.copy()
                sigmoidValuesForEveryLayer.append(layer_values.copy())
                if isBias:
                    values.append(1.0)

            if isBias:
                values = values[:-1]

            irisType = classes[data.iloc[era, 4]]


            allDeltasLayer = []
            outPutLayer = []

            for value in range(layers[-1]):
                # Delta dla warstwy wyjsciowej = Poprawna opd - faktyczna odp * pochodna funkci aktywaacji(faktyczna odp)
                outPutLayer.append((irisType[value] - values[value]) * sigmoidDerivative(values[value]))
            allDeltasLayer.append(outPutLayer)

            for layerIdx in range(len(layers) - 2, 0, -1):
                deltasLayer = []
                for currNeuron in range((layers[layerIdx])):

                    deltasSum = 0
                    for nextNeuron in range(len(allDeltasLayer[-1])):
                        deltasSum += weights[layerIdx][nextNeuron][currNeuron] * allDeltasLayer[-1][nextNeuron]

                    derivative = sigmoidDerivative(sigmoidValuesForEveryLayer[layerIdx-1][currNeuron])
                    delta = deltasSum * derivative
                    deltasLayer.append(delta)

                allDeltasLayer.append(deltasLayer)

            allDeltasLayer = allDeltasLayer[::-1]

            for layerInd in range(len(layers) - 1):

                for currNeuron in range(len(inputsForEveryLayer[layerInd])):

                    for nextNeuron in range(len(allDeltasLayer[layerInd])):
                        gradient = learningSpeed * allDeltasLayer[layerInd][nextNeuron] * inputsForEveryLayer[layerInd][
                            currNeuron]
                        if useMomentum:
                            step = gradient + (momentum[layerInd][nextNeuron][currNeuron] * momentumCoef)
                            weights[layerInd][nextNeuron][currNeuron] += step
                            momentum[layerInd][nextNeuron][currNeuron] = step
                        else:
                            weights[layerInd][nextNeuron][currNeuron] += gradient
            if epoch == maxEpochs - 1:
                print(f"Values: {values}. True answer: {irisType}")


    return weights, allDeltasLayer, values






