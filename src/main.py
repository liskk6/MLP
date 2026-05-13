import pandas as pd
from mlp import MLP


def main():
    col_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
    data = pd.read_csv('../data/iris.txt', header=None, names=col_names)
    #while True:
     #   mode = modeOfOperation()
    print(len(data))
    print(MLP(data, [4,4,3], 100, True, True, False))







if __name__ =="__main__":
    main()


