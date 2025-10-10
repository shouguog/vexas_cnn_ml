import numpy as np
import pandas as pd
data_dir = "../../../../data/vexas_cnn_data/processed/"
data = pd.read_csv(data_dir + 'train.csv')
x_train = data["path"].values
y_train = data.drop(['Unnamed: 0','path', 'group'], axis=1).values
data_test = pd.read_csv(data_dir + 'test.csv')
x_train_test = data_test["path"].values
y_train_test = data_test.drop(['path', 'group'], axis=1).values
###Let us save the dataset
###Let us save the dataset
x_train.tofile("trainDataPath.txt", sep='\n', format='%s')
x_train_test.tofile("testDataPath.txt", sep='\n', format='%s')
