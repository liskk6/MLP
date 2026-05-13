import pandas as pd
import random
import math
def modeOfOperation():

    while True:
        print("Choose mode of operation")
        print("1 - Teaching mode")
        print("2 - Testing mode")
        inp = input("Your choice: ")
        if (inp == '1') or (inp == '2'):
            return int(inp)

def openFile(fileName : str):
    df = pd.read_csv(fileName, sep=',')
    return df

def neuron(inputs):
    wagi = []
    for i in range (len(inputs)):
        wagi.append(random.uniform(-1,1))
    bias = random.uniform(-1,1)

    for i in range(len(inputs)):
        total_suma =+ inputs[i] * wagi[i]

    total_suma =+ bias

    result = funkcja_sigmoidalna(total_suma)

    return result

def funkcja_sigmoidalna(x):
    sig = 1 / (1 + math.exp(x))
    return sig

def warstwa(countNeuron, countSygnal):

    num_neuron = []

    for i in range(countNeuron):
        num_neuron.append(neuron(countSygnal))


