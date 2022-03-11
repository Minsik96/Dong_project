import tensorflow as tf

xData = [1,2,3,4,5,6,7]
yData = [25000, 55000, 75000, 110000, 128000, 155000, 180000]
W = tf.Variable(tf.random.uniform([1],-100,100))
b = tf.Variable(tf.random.uniform([1],-100,100))
print(W, b)

for i in range(10001):

    with tf.GradientTape() as tape :
        H = W * xData + b
        cost = tf.reduce_mean(tf.square(H - yData))
        W_grad, b_grad = tape.gradient(cost,[W,b])

        learning_rate = 0.001
        W.assign_sub(W_grad * learning_rate)
        b.assign_sub(b_grad * learning_rate)

    if i % 1000 == 0 :
        print("{}회 학습을 완료.".format(i))
        print(W, b)

def predict_(t):
    H_pre = W * t + b
    return H_pre

print("8시간 일했을경우 받는 돈 : ",predict_(8))