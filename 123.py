import numpy as np
from pandas.io.parsers import read_csv

x = np.array([[1,2,3],[4,5,6]])

print(x)
print(x.shape)




datas = read_csv("price_predict_project/price data.csv")

xy = np.array(datas, dtype="float")
x = xy[:,1:-1]
y = xy[:, [-1]]


print(x)
print(x.shape)