import pandas as pd
import os
from mlp import MLP,MLP_test
from functions import getMLP,saveMLP
from autoencoder import autoencoder



def main():

    #col_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
    data = pd.read_csv('../data/iris.txt', header=None)
    colsToNorm = data.columns[:-1]
    dataNorm = data.copy()
    dataNorm[colsToNorm] = (data[colsToNorm] - data[colsToNorm].min()) / (
        data[colsToNorm].max() - data[colsToNorm].min())

    MLPSettings = [[4,4,3], 1000 , True, True, 0.2, 0.9, f"statsTraining", "default"]
    MLPSettingsStr = ['Layers:', "Epochs:", "Is random:", "Is bias:", "Learning speed:", "Momentum coefficient:", "Logs file name:", "File to save settings name:"]
    weights = None

    choice = -1
    while choice != 0:
        print(f"--Main menu--\nChoose action: \n[1] Train MLP\n[2] Test MLP\n[3] Autoencoder "
              f"\n[4] Show/Change MLP parameters\n[5] Load settings from file\n[6] Save settings to file\n"
              f"[0] Exit")


        choice = int(input("Your choice: "))
        os.system("cls")
        if choice < 0 or choice > 10:
            print(f"!!!Bad value!!!")
            continue
        else:
            if choice ==  1:
                print("Training started")
                weights = MLP(dataNorm, MLPSettings[0], MLPSettings[1], MLPSettings[2], MLPSettings[3], MLPSettings[4],
                    MLPSettings[5], MLPSettings[6])
            elif choice == 2:
                if weights is None:
                    print("You need to train model")
                    continue
                row = int(input("Choose a row to test (1-150): "))
                if row >= 1 and row <= len(data):
                    MLP_test(dataNorm, row-1,MLPSettings[0], MLPSettings[3], weights)
                else:
                    print("There is no that row")
            elif choice == 3:
                autoencoder(MLPSettings[1],MLPSettings[2],MLPSettings[3], MLPSettings[4], MLPSettings[5])
            elif choice == 4:
                changeChoice = -1
                while changeChoice != 0:
                    print("Current settings:")
                    for i in range(len(MLPSettings)):
                        print(f"{i+1}. {MLPSettingsStr[i]} - {MLPSettings[i]}")
                    changeChoice = int(input(f"0. Exit\nEnter number you want to change 1 - {len(MLPSettings)};\nYour choice: "))

                    if changeChoice == 1:
                        newLayers = []
                        numHiddenLayers = int(input("Enter number of hidden the layers: "))
                        if numHiddenLayers > 0:
                            for i in range(numHiddenLayers):
                                newLayers.append((int(input(f"Enter the number of neurons for hidden layer {i+1} :"))))

                            finalNewLayers = [4]
                            for j in range(len(newLayers)):
                                if newLayers[j] > 0:
                                    finalNewLayers.append(newLayers[j])
                                else:
                                    continue

                            if len(newLayers) + 1 ==  len(finalNewLayers):
                                finalNewLayers.append(3)
                                MLPSettings[0] = finalNewLayers
                                print("Layers were changed")
                            else:
                                print(f"Layers were not changed!")
                        else:
                            print("Error in length!")


                    elif changeChoice == 2:
                        newEpochs = int(input("Enter number of epochs you want: "))
                        if newEpochs > 0:
                            MLPSettings[1] = newEpochs
                            print("Number of epochs was changed")
                        else:
                            print("Number of epochs was not changed")

                    elif changeChoice == 3:
                        print("Random: 0 - Off; 1 - On ")
                        isRandom = int(input("Your choice"))
                        if isRandom == 0:
                            MLPSettings[2] = False
                            print("Random was turned off")
                        elif isRandom == 1:
                            MLPSettings[2] = True
                            print("Random was turned on")
                        else:
                            print("Random was not changed")

                    elif changeChoice == 4:
                        print("Bias: 0 - Off; 1 - On ")
                        isBias = int(input("Your choice"))
                        if isBias == 0:
                            MLPSettings[3] = False
                            print("Bias was turned off")
                        elif isBias == 1:
                            MLPSettings[3] = True
                            print("Bias was turned on")
                        else:
                            print("Bias was not changed")

                    elif changeChoice == 5:
                        newLearningSpeed = float(input(("Enter new learning speed:")))
                        if newLearningSpeed > 0.0 and newLearningSpeed < 1.0:
                            MLPSettings[4] = newLearningSpeed
                            print("Learning speed was changed")
                        else:
                            print("Learning speed was not changed")

                    elif changeChoice == 6:
                        newMomentumCoef = float(input(("Enter new momentum coefficient:")))
                        if newMomentumCoef > 0.0 and newMomentumCoef < 1.0:
                            MLPSettings[5] = newMomentumCoef
                            print("Momentum coefficient was changed")
                        else:
                            print("Momentum coefficient was not changed")

                    elif changeChoice == 7:
                        MLPSettings[6] = input("Enter the new log file name: ")
                    elif changeChoice == 8:
                        MLPSettings[7] = input("Enter the new file settings name: ")
                    os.system('cls')
            elif choice == 5:
                inpFileName = input("Enter file name")
                path = f'../saves/{inpFileName}.txt'
                MLPSettings[0], MLPSettings[1], MLPSettings[2], MLPSettings[3], MLPSettings[4],MLPSettings[5], weights =  getMLP(path)
                print("Settings was copied")
            elif choice == 6:
                saveMLP(MLPSettings[7], MLPSettings[0],MLPSettings[1],MLPSettings[2], MLPSettings[3], MLPSettings[4],MLPSettings[5], weights)
                print("Settings will be saved after program ends")


















    #l1 = MLP(dataNorm, [4,10,3], 1000, True, True, 0.005, 0.9, True)
    #print(getMLP('../saves/saveTraining.txt'))
    #print(MLP_test(data, 80, [4,10,3], True, l1 ))








if __name__ =="__main__":
    main()


