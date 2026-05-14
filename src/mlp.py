import pandas as pd
import copy
import random
from mlp_functions import getValues,generateWeights,generateIndexToTrain, sigmoid, sigmoidDerivative
def MLP(data : pd.DataFrame,layers: list, maxEra: int, isRandom : bool, isBias: bool, isMomentum , learningSpeed: float):

    #Sprawdza czy dane wprowadzono poprawnie
    if (maxEra > len(data) - 10) or (maxEra < 0):
        return None

    #Lista z indeksami do terningu z uwzglednieniem losowosci
    trainIndices = generateIndexToTrain(isRandom, len(data), maxEra)

    #Wagi generowane przy uzyciu funkcji
    weights = generateWeights(layers, isBias)

    classes = {"Iris-setosa" : [1,0,0], "Iris-versicolor" : [0,1,0], "Iris-virginica" : [0,0,1]}

    #Petla dla kazdej epoki
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




        deltasForEveryLayer = []

        output_results = sigmoidValuesForEveryLayer[-1]
        allDeltasLayer = []
        outPutLayer = []
        for value in range(len(layers)):

            # Delta dla warstwy wyjsciowej = Poprawna opd - faktyczna odp * pochodna funkci aktywaacji(faktyczna odp)
            outPutLayer.append((irisType[value] - values[value]) * sigmoidDerivative(values[value]))

        allDeltasLayer.append(outPutLayer)

        for layer in range(len(layers) - 2, -1, -1):
            deltasLayer = []
            for neuron in range((layers[layer + 1 ])):

                deltasSum = 0
                for value in range(len(allDeltasLayer[-1])):
                    deltasSum += weights[layer][value][neuron] * allDeltasLayer[-1][value]

                derivative = sigmoidDerivative(sigmoidValuesForEveryLayer[layer-1][neuron])
                delta = deltasSum * derivative
                deltasLayer.append(delta)

            allDeltasLayer.append(deltasLayer)

            #for neuron in range(len(layers[layer])):
             #   for weightIndex in range(len(deltasLayer[-1])):
              #      weights[layer][neuron][weightIndex] =  weights[layer][neuron][weightIndex] * (learningSpeed * deltasForLayer[neuo])






































    return weights, allDeltasLayer






