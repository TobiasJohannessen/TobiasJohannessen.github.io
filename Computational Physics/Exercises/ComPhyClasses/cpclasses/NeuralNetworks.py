import torch
import matplotlib.pyplot as plt
import numpy as np

class NeuralNetwork():

    def __init__(self):
        self.loss_function = torch.nn.MSELoss()
        self.loss = 0


    def fit(self, x, y, epochs=1000, lr=0.01, optimizer = torch.optim.SGD, **kwargs):
        optimizer = optimizer(self.parameters(), lr=lr, **kwargs)
        loss_function = torch.nn.MSELoss()
    

        for epoch in range(epochs):
            optimizer.zero_grad()
            output = self.forward(x)
            loss = loss_function(output, y)
            loss.backward()
            optimizer.step()

        self.loss = loss.item()

    def plot(self,ax, xs, ys, data_label = 'Data', model_label ='', data_color = 'black', model_color = 'red', plot_data = True):
        x = torch.linspace(torch.min(xs), torch.max(xs), 1000)
        y = self.forward(x)
        with torch.no_grad():
            if plot_data:
                ax.scatter(xs, ys, label = data_label, color = data_color)
            ax.plot(x, y, label=f'{model_label} \n Loss = {self.loss:.2f}', color = model_color)
            
        


    def plot_architecture(self, ax):
        for i, layer in enumerate(self.layers):
            ax.text(0, i, f'Layer {i+1}: {layer.in_features} -> {layer.out_features}', fontsize=12)
        ax.axis('off')
    
        
# Only 1 weight

class OneParameter(torch.nn.Module, NeuralNetwork):

    def __init__(self):
        torch.nn.Module.__init__(self)
        NeuralNetwork.__init__(self)
        self.layer1 = torch.nn.Linear(1, 1, bias=False)
        self.layers = [self.layer1]

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
        self.layers = [self.layer1]

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
        self.layers = [self.layer1, self.layer2]

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
        self.layers = [self.layer1,self.activation_function, self.layer2]

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
        self.layers = [self.layer1,self.activation_function, self.layer2]

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
        self.layers = [self.layer1]

    def forward(self, x): # We implement the forward pass
        x = x.view(-1, 2)
        x = self.layer1(x)
        return x
    



class Linear2x2(torch.nn.Module, NeuralNetwork):

    def __init__(self):
        torch.nn.Module.__init__(self)
        NeuralNetwork.__init__(self)
        self.layer1 = torch.nn.Linear(2, 2, bias=True)
        self.layers = [self.layer1]

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
        self.layers = [self.layer1, self.layer2]

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
        self.layers = [self.layer1, self.dropout, self.layer2]

    def forward(self, x): # We implement the forward pass
        x = x.view(-1, 1)
        x = self.layer1(x)
        x = self.dropout(x)
        x = self.layer2(x)
        return x.flatten()

class CustomLinearNetwork(torch.nn.Module,NeuralNetwork):

    def __init__(self, n_hidden = 1, n_neurons = 10, activation = 'tanh', **kwargs):
        torch.nn.Module.__init__(self)
        NeuralNetwork.__init__(self)
        self.n_hidden = n_hidden
        self.n_neurons = n_neurons
        match activation:
            case 'tanh':
                self.activation = torch.nn.Tanh()

            case 'relu':
                self.activation = torch.nn.ReLU()
            
            case 'silu':
                self.activation = torch.nn.SiLU()
            
            case _:
                raise ValueError(f'Activation function {activation} not implemented')

        self.layers = []
        self.layers.append(torch.nn.Linear(1, self.n_neurons))
        self.layers.append(self.activation)
        for i in range(self.n_hidden):
            self.layers.append(torch.nn.Linear(self.n_neurons, self.n_neurons))
            self.layers.append(self.activation)
        self.layers.append(torch.nn.Linear(self.n_neurons, 1))
        self.model = torch.nn.Sequential(*self.layers)

    def forward(self, x):
        x = x.view(-1,1)  
        x = self.model(x)
        x = x.flatten()
        return x

    def forward2(self,x):
        for layer in self.layers[:-1]:
            x = layer(x)
            x = self.activation(x)
        x = self.layers[-1](x)
        return x


    def predict(self, x):
        x = torch.tensor(x, dtype = torch.float32).reshape(-1,1)
        return self.model(x).detach().numpy().flatten()