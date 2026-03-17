import random
import torch
import matplotlib.pyplot as plt

"""
Create a sample dataset here.
sample size = 10000
weight = [2, -3.4]^T
b = 4.2
y = Xw + b + noise
"""
# generate y = Xw + b + noise
def synthetic_data(w, b, num_examples):
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape(-1, 1)

# define real w and b value, also get the features(X) and labels(y)
true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 10000)

# show the linear relationship between features[:,1] and labels
# plt.scatter(features[:,1].detach().numpy(), labels.detach().numpy(), 1)
# plt.show()

# read a batch of dataset randomly
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    # create indices for accessing the dataset
    indices = list(range(num_examples))
    # reads the dataset randomly
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(indices[i: min(i + batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]

batch_size = 10
# data iteration example
# for X, y in data_iter(batch_size, features, labels):
#     print(X, '\n', y)
#     break

# initialize the model parameters
w = torch.normal(0, 0.01, (2,1),requires_grad=True)
b = torch.zeros(1, requires_grad=True)

# define model
def linreg(X, w, b):
    # linear regression
    return torch.matmul(X,w) + b

# define loss function
def squared_loss(y_hat, y):
    # square loss function
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2

# stochastic gradient descent
# update params
def sgd(params, lr, batch_size):
    # small batch size sgd
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad/batch_size
            param.grad.zero_()

# training
# define the hyper param
lr = 0.03
num_epochs = 3
net = linreg
loss = squared_loss

# training process
for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        l = loss(net(X, w, b), y)
        # since the shape of l is (batch_size, 1), not a scalar,
        # we need add up all the elements in loss
        l.sum().backward()
        sgd([w, b], lr, batch_size) # update the params
    with torch.no_grad():
        train_l = loss(net(features, w, b), labels)
        print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')

print(f'w\'s error of estimate: {true_w - w.reshape(true_w.shape)}')
print(f'b\'s error of estimate: {true_b - b}')
