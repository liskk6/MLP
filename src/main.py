import pandas as pd
from mlp import MLP,MLP_test


def main():
    col_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
    data = pd.read_csv('../data/iris.txt', header=None, names=col_names)
    colsToNorm = data.columns[:-1]
    dataNorm = data
    dataNorm[colsToNorm] = (data[colsToNorm] - data[colsToNorm].min()) / (
        data[colsToNorm].max() - data[colsToNorm].min())

    #while True:
        #mode = modeOfOperation()

    l1 = MLP(dataNorm, [4,4,4,3], 1000, True, True, 0.005, 0.9)
    print(MLP_test(data, 80, [4,4,4,3], True, l1 ))








if __name__ =="__main__":
    main()


