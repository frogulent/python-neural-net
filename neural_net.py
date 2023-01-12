import numpy as np
import math
import random
rng = np.random.default_rng(42)

def sigmoid(x):
    if x < -36:
        x = -36
    return (1 / (1 + math.exp(-x)))

class NeuralNet:
    def __init__(self, weights=None, biases=None, act=sigmoid):
        self.weights = weights
        self.biases = biases
        self.act = act   #is this evil? activation function
        # also should have derivative of the activation function
        
    def test(self): #prints weights n biases
        num_layers = len(self.weights)
        for x in range(num_layers):
            layer = self.weights[x]
            print(layer, self.biases[x])
    
    def create_empty_weights(self, size): #size is a list of how many nodes are in each layer
        weights = [0.0]*(len(size)-1)
        biases = [0.0]*(len(size)-1)
        for x in range(0, len(size)-1):
            weights[x] = [[0.0]*size[x+1]]*size[x]
            biases[x] = [0.0]*size[x+1]
        self.weights = weights
        self.biases = biases
        
    def run(self, inputs):
        input_layer = np.array(inputs)
        nodes = [inputs]
        for i,w in enumerate(self.weights):
            hidden_layer = ([input_layer] @ w) + self.biases[i]
            input_layer = list(map(self.act, hidden_layer[0]))
            nodes.append(input_layer)
        self.nodes = nodes
        return nodes
    
    def mutate_weights(self, mutation_rate=1): #adds a number proportional to mutation_rate to each weight
        for i, w in enumerate(self.weights):
            weights = np.array(w)
            weights += (rng.random(weights.shape) - .5) * mutation_rate
            self.weights[i] = weights
            
            bias = np.array(self.biases[i])
            bias += (rng.random(bias.shape) - .5) * mutation_rate
            self.biases[i] = bias
    
    def add_input(self):
        new_shape = list(self.weights[0].shape) + np.array([1,0])
        self.weights[0].resize(new_shape)
        
    def add_output(self):
        s = self.weights[-1].shape[1]
        self.weights[-1] = np.column_stack((self.weights[-1], [0]*s))
    
    def delete_input(self, i):
        self.weights[0] = np.delete(self.weights[0], i, axis=0)
    
    def delete_output(self, i):
        self.weights[-1] = np.delete(self.weights[-1], i, axis=1)
        
    # def add_hidden(self, layer):
        
    def train(self, target, learning_rate): #backpropagation
        nodes = self.nodes
        for x in reversed(range(1, len(nodes))):
            output = np.array(nodes[x])
            input = np.array([nodes[x-1]])
            weights = self.weights[x-1]
            if x == len(nodes)-1:
                err = target - output
            d = output * (1 - output) #sigmoid derivative
            gradient = err * d * learning_rate
            self.biases[x-1] += gradient
            self.weights[x-1] += input.T @ np.array([gradient])
            
            err = err @ weights.T 

nn = NeuralNet()
nn.create_empty_weights((2,2,1))
nn.mutate_weights()



xor = [[1,1],
       [1,0],
       [0,1],
       [0,0]]
       
ans = [[0],
       [1],
       [1],
       [0]]
       

lr = .5
for x in range(50000):
    c = random.randint(0,3)
    nn.run(xor[c])
    nn.train(ans[c], lr)

    
for x in xor:
    nn.run(x)
    print(nn.nodes)
