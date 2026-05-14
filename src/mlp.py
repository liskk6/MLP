import pandas as pd
import copy

from mlp_functions import getValues,generateWeights,generateIndexToTrain, sigmoid, sigmoidDerivative
def MLP(data : pd.DataFrame,layers: list, maxEra: int, isRandom : bool, isBias: bool, isMomentum , learningSpeed: float):

    #Sprawdza czy dane wprowadzono poprawnie
    if (maxEra > len(data) - 10) or (maxEra < 0):
        return None
    classes = {"Iris-setosa": [1, 0, 0], "Iris-versicolor": [0, 1, 0], "Iris-virginica": [0, 0, 1]}

    #Wagi generowane przy uzyciu funkcji
    weights = generateWeights(layers, isBias)

    # Lista z indeksami do terningu z uwzglednieniem losowosci
    trainIndices = generateIndexToTrain(isRandom, len(data), maxEra)

    #if isMomentum:


    epochs = 5000
    for epoch in range(epochs):
        #Petla dla kazdej epoki

        if isRandom:
            trainIndices = generateIndexToTrain(isRandom, len(data), maxEra)

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
                        weights[layerInd][nextNeuron][currNeuron] =  weights[layerInd][nextNeuron][currNeuron] + (learningSpeed *  allDeltasLayer[layerInd][nextNeuron] * inputsForEveryLayer[layerInd][currNeuron])
            if epoch == epochs - 1:
                print(f"Values: {values}. True answer: {irisType}")


    return weights, allDeltasLayer, values






