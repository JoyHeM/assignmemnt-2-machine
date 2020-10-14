import numpy as np
import time
import random
import matplotlib.pyplot as plt

def sigmoid(x):
    # activation function, sigmoid
    return 1 / (1 + np.exp(-x))


def mse_loss(pred, label):
    # mean squ error
    return ((label - pred) ** 2) / 2


def sigmoid_grad(x):
    # derivative of sigmoid
    return sigmoid(x) * (1 - sigmoid(x))


def load_data(path):
    with open(path, "r") as file:
        lines = file.readlines()
        x = []
        y = []

        for l in lines:
            info = l.split(",")
            x.append(list(map(float, info[0:4])))

            if info[4] == "Iris-setosa\n":
                y.append(0)
            elif info[4] == "Iris-versicolor\n":
                y.append(1)
            else:
                y.append(2)

    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    y = y.reshape(y.shape[0], 1)

    y_axit = x.transpose()
    x_axis = [ i for i in range(len(y_axit[0])) ]

    for i in range(4):
        plt.plot(x_axis, y_axit[i], 'r', label='var'+str(i))
        # print('var'+str(i),np.max(y_axit[i]),np.min(y_axit[i]))

    plt.grid()
    plt.show()

    x_normed = x / x.max(axis=0)
    print(x_normed)
    return x_normed, y


class FC_Layer():
    """
    full connection layer:  y = wx + b
    """
    def __init__(self, input_d, output_d):

        self.W = np.random.normal(0, 0.1, (input_d, output_d))
        self.b = np.random.normal(0, 0.1, (1, output_d))

        # self.grad_W = np.zeros((input_d, output_d))
        # self.grad_b = np.zeros((1, output_d))

        self.net = None
        self.out = None

    def forward(self, X):
        self.net = np.matmul(X, self.W) #+ self.b
        self.out = sigmoid(self.net)
        return self.out

    def backprop(self, X, d_loss, lr):

        out2net = self.out * (1 - self.out)
        net2weight = np.sum(X)
        bp_w = d_loss * out2net * net2weight
        # update weight
        self.W = self.W - lr * bp_w

        return bp_w


if __name__ == "__main__":

    total_epoches = 50
    lr = 0.1

    # 0. load and pre-process data
    xs, ys = load_data("irisdataset.data")

    # 1. create model
    input_layer = FC_Layer(4, 8)
    hidden_layer = FC_Layer(8, 8)
    output_layer = FC_Layer(8, 1)


    # 2. train forword
    for e in range(total_epoches):
        # shuffle data
        c = list(zip(xs, ys))
        random.shuffle(c)
        xs, ys = zip(*c)

        correct_count = 0
        loss = 0

        tp = 0
        fp = 0
        fn = 0
        tn = 0

        for x, y in zip(xs, ys):

            out_layer1 = input_layer.forward(x)
            out_layer2 = hidden_layer.forward(out_layer1)
            out_layer3 = output_layer.forward(out_layer2)

            pred = out_layer3.round()

            # print(pred)
            # calculate accurate and loss
            loss += mse_loss(out_layer3, y)

            if pred == y:
                correct_count += 1

            if pred == y and y == 1:
                tp += 1

            elif pred == y and y == 0:
                tn += 1

            elif pred != y and y == 1:
                fp += 1

            elif pred != y and y == 0:
                fn += 1

            # back-prop
            d_loss = -(y - out_layer3)
            bp_loss3 = output_layer.backprop(x, d_loss, lr)
            bp_loss2 = hidden_layer.backprop(x, bp_loss3, lr)
            bp_loss1 = input_layer.backprop(x, bp_loss2, lr)

            # print(bp_loss2)
            # print(bp_loss1)
        precision = round( tp / (tp + fp) , 2)
        recall = round( tp / (tp + fn) , 2)
        print(precision,recall)
        loss = loss / 100
        print("Epoch: " + str(e) + "  " + str(correct_count) + "% correct with loss of : " + str(loss))
        time.sleep(0.2)
    print("End")