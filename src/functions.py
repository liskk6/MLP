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
    result = []
    with open(path, 'r', encoding='utf-8') as file:
        for line in file:
            result.append(ast.literal_eval(line.strip()))

    return result


