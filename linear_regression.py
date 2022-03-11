import tensorflow as tf

xData = [1,2,3,4,5,6,7]
yData = [25000, 55000, 75000, 110000, 128000, 155000, 180000]
W = tf.Variable(tf.random.uniform([1],-100,100))
b = tf.Variable(tf.random.uniform([1],-100,100))


for i in range(5001):
    with tf.GradientTape() as tape:
        H = W * xData + b
        cost = tf.reduce_mean(tf.square(H - yData))

    grads = tape.gradient(cost,[W,b])

    W.assign_sub(0.0001 * grads[0])
    b.assign_sub(0.0001 * grads[1])

    if i % 500 == 0 :
        print("{0}회 트레이닝 완료 : weight = {1} , bias = {2}".format(i, W, b))

def run_linear(x):
    H_pre = W * x + b
    return H_pre

print("8시간 일하면 벌수있는 결과 예측값 : ", run_linear(8))
print("9시간 일하면 벌수있는 결과 예측값 : ", run_linear(9))