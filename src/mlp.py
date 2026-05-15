import pandas as pd
import copy
import random
import numpy as np

from mlp_functions import getValues,generateWeights,generateIndexToTrain, sigmoid, sigmoidDerivative


def MLP(data : pd.DataFrame, layers: list, maxEpochs: int, isRandom : bool, isBias: bool, learningSpeed: float, momentumCoef : float, weightsParam=None):

    #Sprawdza czy dane wprowadzono poprawnie
    if (maxEpochs < 0) or (maxEpochs < 0) or (learningSpeed < 0):
        return None
    #Slownik z poprawnymi odpowiedziami
    classes = {"Iris-setosa": [1, 0, 0], "Iris-versicolor": [0, 1, 0], "Iris-virginica": [0, 0, 1]}

    #Wagi generowane przy uzyciu funkcji lub przyjmowane z funkcji
    if weightsParam is not None:
        weights = weightsParam
    else:
        weights = generateWeights(layers, isBias, "weights")

    #Lista zer dla momentum
    momentum = generateWeights(layers, isBias, "momentum")

    #Lista indeksow od 0 do 150)
    trainIndices =  [i for i in range(0, 150)]

    #Warunek uzycia momentum. Jesli wspolczynnik < 0 momentum nie bedzie uwzgledniony
    useMomentum = momentumCoef > 0.0




    for epoch in range(maxEpochs):
        #Macierz pomylek
        confusionMatrix =[[0 for _ in range(layers[-1])] for _ in range(layers[-1])]

        #Jesli wlaczona losowosc
        if isRandom:
            trainIndices = random.sample(trainIndices,150)

        for era in trainIndices:
            #Pobrane dane
            originalValues = getValues(data, era, layers[0])

            #Kopia ktora potem bedzie sie zmieniala
            values = copy.deepcopy(originalValues)

            #Wszystkie wartosci funkcji aktywacji
            sigmoidValuesForEveryLayer = []

            #Wszystkie dane podawane w przod
            inputsForEveryLayer = []

            #Jesli bias jest, to dodajemy 1 no koncu zeby potem po prostu
            if isBias:
                values.append(1.0)

            for layer in range(len(layers) - 1):
                #Dodajemy kopie pobranych wartosci
                inputsForEveryLayer.append(values.copy())

                #Lista przechowujaca wartosci ten warstwy po funkcji aktywacji
                layer_values = []

                for neuron in range(len(weights[layer])):
                    neuronSum = 0
                    #Lista z wagami wzgledem neurone do wszystkich nastepnych neuronow
                    neuronWeights = weights[layer][neuron]
                    for valueIndex in range(len(values)):
                        neuronSum += values[valueIndex] * neuronWeights[valueIndex]

                    layer_values.append(sigmoid(neuronSum))

                #Kopjujemy wyliczone wartosci
                values = layer_values.copy()
                sigmoidValuesForEveryLayer.append(layer_values.copy())

                if isBias:
                    values.append(1.0)
            #Po obliczeniu wartosci usuwamy bias
            if isBias:
                values = values[:-1]

            #Pobiweramy poprawna odpowiedz
            irisType = classes[data.iloc[era, 4]]

            #Lista ze wszytkimi deltami
            allDeltasLayer = []

            #Lista delt dla warstwy wyjsciowejew
            outPutLayer = []

            for value in range(layers[-1]):
                # Delta dla warstwy wyjsciowej = Poprawna opd - faktyczna odp * pochodna funkci aktywaacji(faktyczna odp)
                outPutLayer.append((irisType[value] - values[value]) * sigmoidDerivative(values[value]))
            allDeltasLayer.append(outPutLayer)

            #Oblicza delte tylko warstw ukrytych
            for layerIdx in range(len(layers) - 2, 0, -1):
                deltasLayer = []
                for currNeuron in range((layers[layerIdx])):

                    deltasSum = 0
                    for nextNeuron in range(len(allDeltasLayer[-1])):
                        # Wagi razy wartosc delty
                        deltasSum += weights[layerIdx][nextNeuron][currNeuron] * allDeltasLayer[-1][nextNeuron]

                    #Wylicza pochodna
                    derivative = sigmoidDerivative(sigmoidValuesForEveryLayer[layerIdx-1][currNeuron])
                    #Delta
                    delta = deltasSum * derivative
                    deltasLayer.append(delta)

                allDeltasLayer.append(deltasLayer)

            #Odwraca liste
            allDeltasLayer = allDeltasLayer[::-1]

            #Pobieramy indeks poprawnej odpowiedzi
            correctAnswer = irisType.index(max(irisType))
            #pobieramy indeks wyliczonej odpowiedzi
            mlpAnswer = values.index(max(values))

            #Dodajemy do macierzy pomylek
            confusionMatrix[correctAnswer][mlpAnswer] += 1




            #Oblicza nowe wagi i ewentualnie momentum
            for layerInd in range(len(layers) - 1):

                for currNeuron in range(len(inputsForEveryLayer[layerInd])):

                    for nextNeuron in range(len(allDeltasLayer[layerInd])):
                        #Wartosc gradientu
                        gradient = learningSpeed * allDeltasLayer[layerInd][nextNeuron] * inputsForEveryLayer[layerInd][
                            currNeuron]
                        #Uwzglednia momentum
                        if useMomentum:
                            step = gradient + (momentum[layerInd][nextNeuron][currNeuron] * momentumCoef)
                            weights[layerInd][nextNeuron][currNeuron] += step
                            momentum[layerInd][nextNeuron][currNeuron] = step
                        else:
                            weights[layerInd][nextNeuron][currNeuron] += gradient
        correctSum = 0
        for i in range(len(confusionMatrix)):
            correctSum += confusionMatrix[i][i]
        if epoch % 100 == 0 :
            print(f"Epoch - {epoch}/{maxEpochs}. Matrix -{confusionMatrix} - {correctSum}/150 {round(correctSum/150*100, 3)}%")



    return weights, confusionMatrix






