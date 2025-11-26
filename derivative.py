import numpy as np
import matplotlib.pyplot as plt

true_val = 1


x = np.array([1., 0., -1.])
w = np.array([0.2, 0.8, -0.5])
b = 0.5

learning_rate = 0.001

loss_list = []

# forward pass
def forward(x, w, b):
    total = np.dot(x, w) + b
    return total

def ReLU(x):
    return np.maximum(0, x)


for i in range(100):
    forward_output = forward(x, w, b)
    output = ReLU(forward_output)

    #print("output:", output)


    loss = (output - true_val) ** 2
    loss_list.append(loss)

    print("loss:", loss)



    dl = 2*(output - true_val)
    dReLU = np.where(forward_output > 0, 1, 0)
    dforward = dl * dReLU
    dw = np.array(x) * dforward
    db = dforward
    #print("dw:", dw)
    #print("db:", db)

    # update weights and bias
    # decrease learning rate for stability every 10 iterations
    if i % 10 == 0:
         learning_rate = learning_rate * 0.9 if i != 0 else 0.01    

    w = w - learning_rate * dw
    b = b - learning_rate * db

print("Updated weights:", w)
print("Updated bias:", b)


plt.plot(loss_list)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Loss over Iterations')
plt.show()