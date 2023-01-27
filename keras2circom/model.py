# Read keras model into list of parameters like op, input, output, weight, bias
from __future__ import annotations
from dataclasses import dataclass
import typing
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Layer as KerasLayer
import numpy as np
from difflib import SequenceMatcher

supported_ops = [
    'Activation',
    'AveragePooling2D',
    'BatchNormalization',
    'Conv2D',
    'Dense',
    'Flatten',
    'GlobalAveragePooling2D',
    'GlobalMaxPooling2D',
    'Lambda', # only for polynomial activation in the form of `Lambda(lambda x: x**2+x)`
    'MaxPooling2D',
    'ReLU',
]

argmax_ops = [
    'Activation',
    'Softmax',
    'Dense',
]

skip_ops = [
    'Dropout',
    'InputLayer',
]

poly_activation = '4wEAAAAAAAAAAAAAAAEAAAACAAAAQwAAAHMMAAAAfABkARMAfAAXAFMAKQJO6QIAAACpACkB2gF4\ncgIAAAByAgAAAHpOL3Zhci9mb2xkZXJzL2d0L3NnM3Y4cmQxM2w1Mmp4OTFtZmJnemJmYzAwMDBn\nbi9UL2lweWtlcm5lbF8xNTU3MS8yMTc2NzAzOTE5LnB52gg8bGFtYmRhPggAAADzAAAAAA==\n'

# read each layer in a model and convert it to a class called Layer
@dataclass
class Layer:
    ''' A single layer in a Keras model. '''
    op: str
    name: str
    input: typing.List[int]
    output: typing.List[int]
    config: typing.Dict[str, typing.Any]
    weights: typing.List[np.ndarray]

    def __init__(self, layer: KerasLayer):
        self.op = layer.__class__.__name__
        self.name = layer.name
        self.input = layer.input_shape[1:]
        self.output = layer.output_shape[1:]
        self.config = layer.get_config()
        self.weights = layer.get_weights()

class Model:
    layers: typing.List[Layer]

    def __init__(self, filename: str, raw: bool = False):
        ''' Load a Keras model from a file. '''
        model = load_model(filename)
        
        self.layers = [Layer(layer) for layer in model.layers if layer.__class__.__name__ not in skip_ops]
        
        end = len(self.layers)

        # check if the last layer is argmax
        if not raw:
            if self.layers[-1].op not in argmax_ops:
                raise ValueError('Last layer must be Softmax, Activation, or Dense')
            if 'activation' in self.layers[-1].config:
                if self.layers[-1].config['activation'] != 'softmax':
                    raise ValueError('Activation of last layer must be softmax')
        else:
            if self.layers[-1].op not in supported_ops+argmax_ops:
                raise ValueError('Last layer must be supported op, Softmax, Activation, or Dense')
        
        for layer in self.layers[:-1]:
            if layer.op not in supported_ops:
                raise ValueError(f'Unsupported op: {layer.op}')
            if layer.op == 'Lambda':
                s = SequenceMatcher(None, layer.config['function'][0], poly_activation)
                if s.ratio() < 0.99:
                    raise ValueError('Unsupported lambda function')
            if 'activation' in layer.config:
                if layer.config['activation'] not in ['linear', 'relu']:
                    print(layer.config['activation'])
                    raise ValueError('Unsupported activation function')