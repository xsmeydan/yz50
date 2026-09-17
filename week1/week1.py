#forward pass

input = [1, 2, 3]
weights = [0.2, 0.5, 0.3]
bias = 1

result = sum(i * w for i, w in zip(input, weights)) + bias
sigmoid = 1 / (1 + 2.71828 ** -result)
print(sigmoid)

# multiple neuron forward pass

weights1 = [0.2, 0.5, 0.3]
weights2 = [0.4, 0.6, 0.8]
weights3 = [0.1, 0.3, 0.2]

bias1 = 3
bias2 = 2
bias3 = 1

result1 = sum(i * w for i, w in zip(input, weights1)) + bias1
result2 = sum(i * w for i, w in zip(input, weights2)) + bias2
result3 = sum(i * w for i, w in zip(input, weights3)) + bias3

sigmoid1 = 1 / (1 + 2.71828 ** -result1)
sigmoid2 = 1 / (1 + 2.71828 ** -result2)
sigmoid3 = 1 / (1 + 2.71828 ** -result3)

print(sigmoid1) # 0.9939
print(sigmoid2) # 0.9975
print(sigmoid3) # 0.9088

# Layer2 

input_l2 = [sigmoid1, sigmoid2, sigmoid3]
weights_l2 = [0.3, 0.5, 0.9]

bias_l2 = 2

result_l2 = sum(i * w for i, w in zip(input_l2, weights_l2)) + bias_l2

sigmoid_l2 = 1 / (1 + 2.71828 ** -result_l2)

print(sigmoid_l2)

# loss function

target = 1 

loss = (target-sigmoid_l2)**2

print(loss)

def loss(target, prediction):
    return (target - prediction) ** 2

loss_value = loss(target, sigmoid_l2)
print(loss_value)

# loss curve

def sigmoid(x):
    return 1 / (1 + 2.71828 ** -x)

def result(input, weights, bias):
    return sum(i * w for i, w in zip(input, weights)) + bias

print("Loss Values")
loss_value_list = []
weight_list = []
for i in range(100):
    weights[0] = i * 0.1
    weight_list.append(weights[0])
    result_value = result(input, weights, bias)
    sigmoid_value = sigmoid(result_value)
    loss_value = loss(target, sigmoid_value)
    loss_value_list.append(loss_value)
    print(loss_value)

import matplotlib.pyplot as plt

plt.plot(weight_list, loss_value_list)
plt.xlabel('Weight[0]')
plt.ylabel('Loss')
plt.title('Loss Curve')
# plt.show()
# plt.pause(10)
# plt.close()

# gradient descent
weights[0] = 0.0
learning_rate = 0.01
h = 0.0001

for i in range(100):
    loss_value = loss(target, sigmoid(result(input, weights, bias)))
    weights[0] = weights[0] + h
    new_loss_value = loss(target, sigmoid(result(input, weights, bias)))
    derivative = (new_loss_value - loss_value) / h
    weights[0] = weights[0] - h
    weights[0] = weights[0] - learning_rate * derivative
    print(i, "weight:", weights[0], "loss:", loss_value, "derivative:", derivative)
print("weight[0]", weights[0])
