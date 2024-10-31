import torch
import matplotlib.pyplot as plt
import numpy as np

class NeuralNetwork():

    def __init__(self):
        pass


    def train_model(self, x, y, epochs=1000, lr=0.01, optimizer = torch.optim.SGD):
        optimizer = optimizer(self.parameters(), lr=lr)
        loss_function = torch.nn.MSELoss()
    

        for epoch in range(epochs):
            optimizer.zero_grad()
            output = self.forward(x)
            loss = loss_function(output, y)
            loss.backward()
            optimizer.step()

    def plot(self,ax, x_low, x_high, label ='', color = 'C0'):
        x = torch.linspace(x_low, x_high, 1000)
        y = self.forward(x)
        ax.plot(x, y.detach().numpy(), label='', color = color)

    
        
# Only 1 weight

class OneParameter(torch.nn.Module, NeuralNetwork):

    def __init__(self):
        torch.nn.Module.__init__(self)
        NeuralNetwork.__init__(self)
        self.layer1 = torch.nn.Linear(1, 1, bias=False)

    def forward(self, x): # We implement the forward pass
        x = x.view(-1, 1)
        x = self.layer1(x)
        return x.flatten()


# Linear model with 1 weight and 1 bias
class LinearModel(torch.nn.Module, NeuralNetwork):

    def __init__(self):
        torch.nn.Module.__init__(self)
        NeuralNetwork.__init__(self)
        self.layer1 = torch.nn.Linear(1, 1, bias=True)
        self.layers = [self.layer1]

    def forward(self, x): # We implement the forward pass
        x = x.view(-1, 1)
        x = self.layer1(x)
        return x.flatten()

# Linear model with 1 weight and 1 bias and ReLU activation function
class LinearModelWithReLU(torch.nn.Module, NeuralNetwork):

    def __init__(self):
        torch.nn.Module.__init__(self)
        NeuralNetwork.__init__(self)
        self.layer1 = torch.nn.Linear(1, 1, bias=True)

    def forward(self, x): # We implement the forward pass
        x = x.view(-1, 1)
        x = self.layer1(x)
        x = torch.relu(x)
        return x.flatten()


class TwoLayerWithReLU(torch.nn.Module, NeuralNetwork):

    def __init__(self):
        torch.nn.Module.__init__(self)
        NeuralNetwork.__init__(self)
        self.layer1 = torch.nn.Linear(1, 8, bias=True)
        self.layer2 = torch.nn.Linear(8, 1, bias=True)

    def forward(self, x): # We implement the forward pass
        x = x.view(-1, 1)
        x = self.layer1(x)
        x = torch.relu(x)
        x = self.layer2(x)
        return x.flatten()


class TwoLayerCustomActivation(torch.nn.Module, NeuralNetwork):
    def __init__(self, activation_function):
        torch.nn.Module.__init__(self)
        NeuralNetwork.__init__(self)
        self.layer1 = torch.nn.Linear(1, 1, bias=True)
        self.layer2 = torch.nn.Linear(1, 1, bias=True)
        self.activation_function = activation_function

    def forward(self, x): # We implement the forward pass
        x = x.view(-1, 1)
        x = self.layer1(x)
        x = self.activation_function(x)
        x = self.layer2(x)
        return x.flatten()

class Two_8_LayerCustomActivation(torch.nn.Module, NeuralNetwork):

    def __init__(self, activation_function):
        torch.nn.Module.__init__(self)
        NeuralNetwork.__init__(self)
        self.layer1 = torch.nn.Linear(1, 8, bias=True)
        self.layer2 = torch.nn.Linear(8, 1, bias=True)
        self.activation_function = activation_function

    def forward(self, x): # We implement the forward pass
        x = x.view(-1, 1)
        x = self.layer1(x)
        x = self.activation_function(x)
        x = self.layer2(x)
        return x.flatten()


class Linear2x2NoBias(torch.nn.Module, NeuralNetwork):

    def __init__(self):
        torch.nn.Module.__init__(self)
        NeuralNetwork.__init__(self)
        self.layer1 = torch.nn.Linear(2, 2, bias=False)


    def forward(self, x): # We implement the forward pass
        x = x.view(-1, 2)
        x = self.layer1(x)
        return x
    



class Linear2x2(torch.nn.Module, NeuralNetwork):

    def __init__(self):
        torch.nn.Module.__init__(self)
        NeuralNetwork.__init__(self)
        self.layer1 = torch.nn.Linear(2, 2, bias=True)


    def forward(self, x): # We implement the forward pass
        x = x.view(-1, 2)
        x = self.layer1(x)
        return x


class TwoLinear2x2(torch.nn.Module, NeuralNetwork):

    def __init__(self):
        torch.nn.Module.__init__(self)
        NeuralNetwork.__init__(self)
        self.layer1 = torch.nn.Linear(2, 2, bias=True)
        self.layer2 = torch.nn.Linear(2, 2, bias=True)


    def forward(self, x): # We implement the forward pass
        x = x.view(-1, 2)
        x = self.layer1(x)
        x = self.layer2(x)
        return x


class Two_8_LayerDropoutLayer(torch.nn.Module, NeuralNetwork):

    def __init__(self, probability = 0.5):
        torch.nn.Module.__init__(self)
        NeuralNetwork.__init__(self)
        self.layer1 = torch.nn.Linear(1, 8, bias=True)
        self.dropout = torch.nn.Dropout(p=probability)
        self.layer2 = torch.nn.Linear(8, 1, bias=True)
        

    def forward(self, x): # We implement the forward pass
        x = x.view(-1, 1)
        x = self.layer1(x)
        x = self.dropout(x)
        x = self.layer2(x)
        return x.flatten()