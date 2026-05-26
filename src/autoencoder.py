from mlp_functions import generateWeights,getValues, forwardPropagation,sigmoidDerivative, generateStatsForSave
from datetime import datetime
import random
import os
import copy
import pandas as pd

def autoencoder(maxEpochs:int ,isRandom:bool, isBias:bool,learningSpeed: float, momentumCoef: float):
    if (maxEpochs <= 0) or (momentumCoef < 0.0) or (learningSpeed < 0.0):
        return None, None
    layers = [4,2,4]
    data = [[1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,1]]
    weights = generateWeights(layers, isBias, 'weights')
    momentum = generateWeights(layers, isBias, "momentum")
    path  = f'../stats/autoencoder.txt'
    trainIndices = [i for i in range(0, len(data))]

    useMomentum = momentumCoef > 0.0

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as file:
        file.write(
            f"MLP settings.\nLayers: {layers}; Epochs: {maxEpochs}; Random: {isRandom}; Bias: {isBias}; Learning speed: {learningSpeed}; Momentum coefficient: {momentumCoef}\n\n")

    training_history = []

    for epoch in range(1, maxEpochs + 1):
        epoch_mse = 0.0
        epoch_results_for_log = []
        autoenkoderResult = []

        if isRandom:
            trainIndices = random.sample(trainIndices, len(data))

        for era in trainIndices:
            # Pobrane dane
            originalValues = data[era]

            values, sigmoidValuesForEveryLayer, inputsForEveryLayer = forwardPropagation(weights, layers, isBias,originalValues)


            autoenkoderResult.append(values)
            allDeltasLayer = []
            outPutLayer = []
            era_mse = 0.0
            individual_errors = []

            for value in range(layers[-1]):
                error = originalValues[value] - values[value]
                individual_errors.append(error)
                era_mse += error ** 2
                # Delta dla warstwy wyjsciowej = Poprawna opd - faktyczna odp * pochodna funkci aktywaacji(faktyczna odp)
                outPutLayer.append(error * sigmoidDerivative(values[value]))



            allDeltasLayer.append(outPutLayer)

            pattern_mse = era_mse / layers[-1]
            epoch_mse += pattern_mse

            epoch_results_for_log.append({
                'original': originalValues,
                'values': values,
                'mse': pattern_mse,
                'ind_errors': individual_errors,
                'weights': copy.deepcopy(weights),
                'sigmoids': copy.deepcopy(sigmoidValuesForEveryLayer)
            })

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

        epoch_mse = epoch_mse / len(data)
        pseudo_accuracy = max(0.0, 1.0 - epoch_mse)
        training_history.append([epoch, epoch_mse, pseudo_accuracy])

        if epoch % 10 == 0:
            print(f"Epoch - {epoch}/{maxEpochs}. MSE: {round(epoch_mse, 5)}")

        if (epoch % 10 == 0) or epoch == maxEpochs or epoch == 1:
            now = datetime.now()
            t = now.strftime("%H:%M:%S")

            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, 'a', encoding='utf-8') as file:
                file.write(f"\n{'-' * 50}\n")
                file.write(f"Epoch {epoch}|{maxEpochs}. Time: {t}\n")


                epoch_results_for_log.sort(key=lambda x: x['original'].index(1))

                for res in epoch_results_for_log:
                    file.write(f"\nOriginal values: {[round(float(i), 5) for i in res['original']]};\n")
                    file.write(f"Output values  : {[round(float(i), 5) for i in res['values']]};\n")
                    file.write(f"MSE: {round(res['mse'], 5):<4};\n")

                    for i in range(len(res['ind_errors'])):
                        file.write(f"Neuron: {i + 1} - Error: {round(res['ind_errors'][i], 5)}\n")
                    roundedWeights = [[[round(float(w), 5) for w in neuron] for neuron in layer] for layer in
                                  res['weights']]
                    file.write(f"Output layer weights: {roundedWeights[-1]}\n")

                    roundedWeightsRev = roundedWeights[::-1]
                    for i in range(len(res['weights']) - 1, 0, -1):
                        file.write(f"Hidden neurons layer weights: {roundedWeightsRev[i]}\n")

                    sigmoidValuesForEveryLayer = [[round(float(w), 5) for w in neuron] for neuron in res['sigmoids']]
                    sigmoidValuesForEveryLayerRev = sigmoidValuesForEveryLayer[::-1]
                    for i in range(len(sigmoidValuesForEveryLayerRev)):
                        file.write(f"Hidden neurons values: {sigmoidValuesForEveryLayerRev[i]}\n")

    history_df = pd.DataFrame(training_history, columns=['Epoch', 'MSE', 'Accuracy'])

    return weights, history_df