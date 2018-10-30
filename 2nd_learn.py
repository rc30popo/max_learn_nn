# -*- Coding: utf-8 -*-

# Numpy
import numpy as np
from numpy.random import *
# Chainer
import chainer
import chainer.links as L
import chainer.functions as F
from chainer import Chain,optimizers,Variable

# Neural Network

class DNN(Chain):
    def __init__(self):
        super(DNN, self).__init__(
            l1 = L.Linear(None,100),
            l2 = L.Linear(None,100),
            l3 = L.Linear(None,100),
            l4 = L.Linear(None,10)
        )
    def forward(self,x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        h3 = F.relu(self.l3(h2))
        h4 = self.l4(h3)
        return h4

n_epoch = 100000
batch_size = 1000
test_size = 100

model = DNN()
# Set optimizer
optimizer = optimizers.Adam()
optimizer.setup(model)

for epoch in range(0,n_epoch):
    sum_loss = 0
    x_train = np.array(rand(batch_size,10),dtype=np.float32)
    t_train = []
    for i in range(0,batch_size):
        sorted_x = np.sort(x_train[i])
        second_index = np.where(x_train[i] == sorted_x[8])[0][0]
        t_train.append(second_index)

    t_train = np.array(t_train,dtype=np.int32)
    x = Variable(x_train)
    t = Variable(t_train)
    y = model.forward(x)
    model.cleargrads()
    loss = F.softmax_cross_entropy(y, t)
    loss.backward()
    optimizer.update()
    print("epoch: {}, mean loss: {}".format(epoch, loss.data))


x_test = np.array(rand(test_size,10),dtype=np.float32)
t_test = []

ok_cnt = 0
for i in range(0,test_size):
    sorted_x = np.sort(x_test[i])
    second_index = np.where(x_test[i] == sorted_x[8])[0][0]

    x = Variable(np.array([x_test[i]],dtype=np.float32))
    y = model.forward(x)
    y = np.argmax(y.data[0])
    if y == second_index:
        ok_cnt += 1
    print("x_test[{}]={}, y={}".format(i,x_test[i],y))

print("Ok = {}/Total = {}".format(ok_cnt,test_size))

