import numpy as np
from activations import Sigmoid
from layers import Affine

class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size):
        I, H, O = input_size, hidden_size, output_size

        W1 = np.random.randn(H, I)
        b1 = np.random.randn(H)
        W2 = np.random.randn(H, O)
        b2 = np.random.randn(O)
    
        self.layers = [
            Affine(W1, b1),
            Sigmoid(),
            Affine(W2, b2)
        ]

        self.params = []
        for layer in self.layers:
            self.params += layer.params
    def predict(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

if __name__=='__main__':
    mdl = TwoLayerNet(4,4,1)
    x = np.random.randn(4, 4)
    print(mdl.predict(x))