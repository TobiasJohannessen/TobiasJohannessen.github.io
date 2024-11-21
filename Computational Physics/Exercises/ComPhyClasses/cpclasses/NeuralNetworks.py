import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset, DataLoader


class NeuralNetwork():

    def __init__(self, loss_function = torch.nn.MSELoss()):
        self.loss_function = loss_function
        self.loss = 0


    def fit(self, x, y, epochs=1000, lr=0.01, optimizer = torch.optim.SGD, **kwargs):
        optimizer = optimizer(self.parameters(), lr=lr, **kwargs)
        loss_function = self.loss_function
    

        for epoch in range(epochs):
            optimizer.zero_grad()
            output = self.forward(x)
            loss = loss_function(output, y)
            loss.backward()
            optimizer.step()

            if epoch % 500 == 0:
                print(f'Epoch {epoch} \n Loss: {loss.item()}')

        self.loss = loss.item()

    def fit_loader(self, training_loader, validation_loader, epochs=1000, lr=0.01, optimizer = torch.optim.SGD, patience = 5, **kwargs):
        self.train_loss = []
        self.val_loss = []

        
        early_stopper = EarlyStopping(patience)

        optimizer = optimizer(self.parameters(), lr=lr, **kwargs)
        loss_function = self.loss_function

        for epoch in range(epochs):
            self.train()
            train_loss = 0.0
            for batch_of_data in training_loader:
                optimizer.zero_grad()
                outputs = self(batch_of_data[0])
                loss = self.loss_function(outputs, batch_of_data[1])
                loss.backward()
                optimizer.step()
                with torch.no_grad():
                    train_loss += loss.item() * batch_of_data[0].size(0) 
            
            train_loss /= len(training_loader.dataset) 

            # Evaluation step
            self.eval()
            val_loss = 0.0
            for batch_of_data in validation_loader:
                with torch.no_grad():
                    outputs = self(batch_of_data[0])
                    val_loss += loss_function(outputs, batch_of_data[1]).item() * batch_of_data[0].size(0) 
            val_loss /= len(validation_loader.dataset)
            if early_stopper is not None:
                early_stopper(val_loss)
               # if early_stopper.early_stop:
                #    break
            
            if epoch % 250 == 0:
                print(f'Epoch {epoch} \n Training loss: {train_loss} \n Validation loss: {val_loss}')
            
                
            self.train_loss.append(train_loss)
            self.val_loss.append(val_loss)


    def plot(self,ax, xs, ys, label = '', **kwargs):
        x = torch.linspace(torch.min(xs), torch.max(xs), 1000)
        y = self.forward(x)
        with torch.no_grad():
            ax.plot(x, y, label=f'{label}' + r'$\mathcal{L}$ =' + f' {self.loss:.2f}', **kwargs)
            
        


    def plot_architecture(self, ax):
        for i, layer in enumerate(self.layers):
            ax.text(0, len(self.layers) - i, f'Layer {i+1}: {layer.in_features} -> {layer.out_features}', fontsize=12)
        ax.axis('off')
    



class CustomDeepNetwork(torch.nn.Module,NeuralNetwork):

    def __init__(self, n_hidden = 1, n_neurons = 10, activation = None, **kwargs):
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
            
            case 'sigmoid':
                self.activation = torch.nn.Sigmoid()

            case 'selu':
                self.activation = torch.nn.SELU()
            
            case None:
                self.activation = torch.nn.Identity()
            
            case default:
                #Allows all torch.nn.Module activation functions to be used
                if isinstance(activation, torch.nn.Module):
                    self.activation = activation

                else:
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

############################################################################
# DATA SETS
############################################################################


class CustomDataset(Dataset):

    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature = self.features[idx]
        label = self.labels[idx]
        return feature, label

    
        
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


class CustomNetwork2(torch.nn.Module, NeuralNetwork):

    def __init__(self):
        torch.nn.Module.__init__(self)
        NeuralNetwork.__init__(self)
        self.inputlayer = torch.nn.Linear(1, 8, bias=True)
        self.layer1 = torch.nn.Linear(8, 8, bias=True)
        self.outputlayer = torch.nn.Linear(8, 1, bias=True)

    def forward(self, x): # We implement the forward pass
        x = x.view(-1, 1)
        x = self.inputlayer(x)
        x = torch.relu(x)
        x = self.layer1(x)
        x = torch.relu(x)
        x = self.outputlayer(x)
        return x.flatten()



############################################################################
# DATA SETS
############################################################################


class CustomDataset(Dataset):

    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature = self.features[idx]
        label = self.labels[idx]
        return feature, label





############################################################################
# EARLY STOPPING
############################################################################

class EarlyStopping():
    def __init__(self, patience=5):
        # patience (int): How many epochs to wait after last time validation loss improved.
        self.patience = patience
        self.best_loss = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0




############################################################################
# CLASSIFICATION
############################################################################


class Classification(torch.nn.Module):
    def __init__(self, n_features=1, n_classes=3, loss_function = torch.nn.CrossEntropyLoss()):
        super().__init__()
        self.fc1 = torch.nn.Linear(n_features, n_classes, bias=True)
        self.n_features = n_features
        self.n_classes = n_classes
        self.loss_function = loss_function
        
    def forward(self,x):
        x = x.view(-1, self.n_features)
        x = self.fc1(x)
        return x

    def predict_label(self, x):

        predicted_weights = self(x)
        predicted_probabilities = torch.nn.functional.softmax(predicted_weights, dim=1)
        predicted_label = torch.argmax(predicted_probabilities, dim=1)
        return predicted_label


    def fit(self, data_list, labels_list, epochs=1000, lr=0.01):
        criterion = torch.nn.CrossEntropyLoss()
        n_epochs = 1000
        optimizer = torch.optim.SGD(self.parameters(), lr=0.01)

        for epoch in range(epochs):
            optimizer.zero_grad()
            predicted_weights = self(data_list)
            labels = torch.tensor(labels_list, dtype = int)
            loss = criterion(predicted_weights, labels)
            loss.backward()
            optimizer.step()

            if epoch % 100 == 0:
                print(f'Epoch {epoch}, loss: {loss.item()}')

        print('Optimization finished with loss: ', loss.item(), '\n') 

    
    def fit_loader(self, training_loader, validation_loader, epochs=1000, lr=0.01, optimizer = torch.optim.SGD, patience = 5, **kwargs):
        self.train_loss = []
        self.val_loss = []

        
        early_stopper = EarlyStopping(patience)

        optimizer = optimizer(self.parameters(), lr=lr, **kwargs)
        loss_function = self.loss_function

        for epoch in range(epochs):
            self.train()
            train_loss = 0.0
            for batch_of_data in training_loader:
                optimizer.zero_grad()
                outputs = self(batch_of_data[0])
                loss = self.loss_function(outputs, batch_of_data[1])
                loss.backward()
                optimizer.step()
                with torch.no_grad():
                    train_loss += loss.item() * batch_of_data[0].size(0) 
            
            train_loss /= len(training_loader.dataset) 

            # Evaluation step
            self.eval()
            val_loss = 0.0
            for batch_of_data in validation_loader:
                with torch.no_grad():
                    outputs = self(batch_of_data[0])
                    val_loss += loss_function(outputs, batch_of_data[1]).item() * batch_of_data[0].size(0) 
            val_loss /= len(validation_loader.dataset)
            if early_stopper is not None:
                early_stopper(val_loss)
               # if early_stopper.early_stop:
                #    break
            
            if epoch % 250 == 0:
                print(f'Epoch {epoch} \n Training loss: {train_loss} \n Validation loss: {val_loss}')
            
                
            self.train_loss.append(train_loss)
            self.val_loss.append(val_loss)

        print('Optimization finished with loss: ', val_loss, '\n')


class Classification_I(Classification):
    def __init__(self, n_features=1, n_classes=3):
        super().__init__(n_features, n_classes)
        self.fc1 = torch.nn.Linear(n_features, n_classes, bias=True)

        
    def forward(self,x):
        x = x.view(-1, self.n_features)
        x = self.fc1(x)
        return x


class Classification_II(Classification):

    def __init__(self, n_features=1, n_classes=3):
        super().__init__(n_features, n_classes)
        self.fc1 = torch.nn.Linear(n_features, 10, bias=True)
        self.fc2 = torch.nn.Linear(10, n_classes, bias=True)

    def forward(self,x):
        x = x.view(-1, self.n_features)
        x = self.fc1(x)
        x = torch.nn.functional.silu(x)
        x = self.fc2(x)
        return x


class Classification_III(Classification):

    def __init__(self, n_features=1, n_classes=3):
        super().__init__(n_features, n_classes)
        self.input_layer = torch.nn.Linear(n_features, 10, bias=True)
        self.hidden_layer = torch.nn.Linear(10, 10, bias=True)
        self.output_layer = torch.nn.Linear(10, n_classes, bias=True)

    def forward(self,x):
        x = x.view(-1, self.n_features)
        x = self.input_layer(x)
        x = torch.nn.functional.silu(x)
        x = self.hidden_layer(x)
        x = torch.nn.functional.silu(x)
        x = self.output_layer(x)
        return x



