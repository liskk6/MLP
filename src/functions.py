import pandas as pd
import os
import math
import matplotlib.pyplot as plt
import ast


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

def saveMLP(fileName: str, layers:list, maxEpochs:int, isRandom:bool, isBias: bool, learningSpeed: float, momentumCoef:float, weights:list ):
    path = f'../saves/{fileName}.txt'
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as file:
        file.write(f"{layers} \n")
        file.write(f"{maxEpochs}\n")
        file.write(f"{isRandom}\n")
        file.write(f"{isBias}\n")
        file.write(f"{learningSpeed}\n")
        file.write(f"{momentumCoef}\n")
        file.write(f"{weights}\n")

def plot_training_history(history_df):
    if history_df is None or history_df.empty:
        print("No training history available!")
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    ax1.plot(history_df['Epoch'], history_df['MSE'], color='red', linewidth=2)
    ax1.set_title('Training Loss (MSE)')
    ax1.set_ylabel('Mean Squared Error')
    ax1.grid(True, linestyle='--', alpha=0.7)

    ax2.plot(history_df['Epoch'], history_df['Accuracy'], color='green', linewidth=2)
    ax2.set_title('Training Accuracy')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy (0 to 1)')
    ax2.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()