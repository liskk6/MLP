import pandas as pd
import random
import math
import ast
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

def getMLP(path:str):

    with open(path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        layers = ast.literal_eval(lines[0].strip())
        maxEpochs = int(lines[1].strip())
        isRandom = lines[2].strip() == 'True'
        isBias = lines[3].strip() == 'True'
        learningSpeed = float(lines[4].strip())
        momentumCoef = float(lines[5].strip())

        weights_str = lines[6].strip().replace("np.float64(", "").replace(")", "")
        weightsParam = ast.literal_eval(weights_str)

        return layers, maxEpochs, isRandom, isBias, learningSpeed, momentumCoef, weightsParam

