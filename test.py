import numpy as np
from pandas.io.parsers import read_csv
from tensorflow.keras.models import Sequential
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import load_model


#데이터 pandas를 통해 읽기.
datas = read_csv("price_predict_project/price data.csv")

xy = np.array(datas, dtype="float")
x = xy[:,1:-1]
y = xy[:, [-1]]



# keras를 통해서 linear regression을 진행
def train_dataset(x,y):
    model = Sequential()
    model.add(Dense(1, input_dim=4, activation='linear'))
    sgd = optimizers.SGD(learning_rate=0.0000005)
    model.compile(optimizer=sgd, loss="mse", metrics="mse")
    model.fit(x,y,epochs=100000, verbose=2)

    model.save("price_predict_project/prection_vegi.h5")
    return

# train_dataset(x,y)

# avg_temp = float(input("평균 온도: "))
# min_temp = float(input("최저 온도: "))
# max_temp = float(input("최고 온도: "))
# rain_fall = float(input("강수량: "))

new_model = load_model("price_predict_project/prection_vegi.h5")
a = ((10,2,20,0),)
arr = np.array(a, dtype="float")
c = arr[0:4]


pr = new_model.predict(c)
print(pr)

# data = ((avg_temp,min_temp,max_temp,rain_fall))
# arr = np.array(data, dtype=np.float)
# xdata = arr[0:4]
#
# pred = new_model.predict(xdata)
# print("예측된 가격: {0}".format(pred))