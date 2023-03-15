from .circom import *
from .model import *

import os

from difflib import SequenceMatcher
poly_activation = '4wEAAAAAAAAAAAAAAAEAAAACAAAAQwAAAHMMAAAAfABkARMAfAAXAFMAKQJO6QIAAACpACkB2gF4\ncgIAAAByAgAAAHpOL3Zhci9mb2xkZXJzL2d0L3NnM3Y4cmQxM2w1Mmp4OTFtZmJnemJmYzAwMDBn\nbi9UL2lweWtlcm5lbF8xNTU3MS8yMTc2NzAzOTE5LnB52gg8bGFtYmRhPggAAADzAAAAAA==\n'

def transpile(filename: str, output_dir: str = 'output', raw: bool = False) -> Circuit:
    ''' Transpile a Keras model to a CIRCOM circuit.'''
    
    model = Model(filename, raw)

    circuit = Circuit()
    for layer in model.layers[:-1]:
        circuit.add_components(transpile_layer(layer))
    
    circuit.add_components(transpile_layer(model.layers[-1], True))

    if raw:
        if circuit.components[-1].template.op_name == 'ArgMax':
            circuit.components.pop()
    # create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    with open(output_dir + '/circuit.json', 'w') as f:
        f.write(circuit.to_json())
    
    with open(output_dir + '/circuit.circom', 'w') as f:
        f.write(circuit.to_circom())
    
    return circuit

def transpile_layer(layer: Layer, last: bool = False) -> typing.List[Component]:
    ''' Transpile a Keras layer to CIRCOM component(s).'''
    if layer.op == 'Activation':
        if layer.config['activation'] == 'softmax':
            if last:
                return transpile_ArgMax(layer)
            raise ValueError('Softmax must be the last layer')
        if layer.config['activation'] == 'relu':
            return transpile_ReLU(layer)
        if layer.config['activation'] == 'linear':
            return []
        raise NotImplementedError(f'Activation {layer.config["activation"]} not implemented')
    
    if layer.op == 'Softmax':
        if last:
            return transpile_ArgMax(layer)
        raise ValueError('Softmax must be the last layer')
    
    if layer.op == 'ReLU':
        return transpile_ReLU(layer)

    if layer.op == 'AveragePooling2D':
        return transpile_AveragePooling2D(layer)
    
    if layer.op == 'BatchNormalization':
        return transpile_BatchNormalization2D(layer)

    if layer.op == 'Conv2D':
        return transpile_Conv2D(layer)
    
    if layer.op == 'Dense':
        return transpile_Dense(layer, last)
        
    if layer.op == 'Flatten':
        return transpile_Flatten2D(layer)

    if layer.op == 'GlobalAveragePooling2D':
        return transpile_GlobalAveragePooling2D(layer)
        
    if layer.op == 'GlobalMaxPooling2D':
        return transpile_GlobalMaxPooling2D(layer)

    if layer.op == 'Lambda':
        s = SequenceMatcher(None, layer.config['function'][0], poly_activation)
        if s.ratio() < 0.95:
            raise ValueError('Only polynomial activation functions are supported')
        return transpile_Poly(layer)
    
    if layer.op == 'MaxPooling2D':
        return transpile_MaxPooling2D(layer)
    
    raise NotImplementedError(f'Layer {layer.op} is not supported yet.')

# TODO: handle scaling
def transpile_ArgMax(layer: Layer) -> typing.List[Component]:
    return [Component(layer.name, templates['ArgMax'], [Signal('in', layer.output)], [Signal('out', (1,))], {'n': layer.output[0]})]

def transpile_ReLU(layer: Layer) -> typing.List[Component]:
    return [Component(layer.name, templates['ReLU'], [Signal('in', layer.output)], [Signal('out', layer.output)])]

def transpile_AveragePooling2D(layer: Layer) -> typing.List[Component]:
    if layer.config['data_format'] != 'channels_last':
        raise NotImplementedError('Only data_format="channels_last" is supported')
    if layer.config['padding'] != 'valid':
        raise NotImplementedError('Only padding="valid" is supported')
    if layer.config['pool_size'][0] != layer.config['pool_size'][1]:
        raise NotImplementedError('Only pool_size[0] == pool_size[1] is supported')
    if layer.config['strides'][0] != layer.config['strides'][1]:
        raise NotImplementedError('Only strides[0] == strides[1] is supported')
    
    return [Component(layer.name, templates['AveragePooling2D'], [Signal('in', layer.input)], [Signal('out', layer.output)],{
        'nRows': layer.input[0],
        'nCols': layer.input[1],
        'nChannels': layer.input[2],
        'poolSize': layer.config['pool_size'][0],
        'strides': layer.config['strides'][0],
        'scaledInvPoolSize': 1/(layer.config['pool_size'][0]**2),
        })]

def transpile_BatchNormalization2D(layer: Layer) -> typing.List[Component]:
    if layer.input.__len__() != 3:
        raise NotImplementedError('Only 2D inputs are supported')
    if layer.config['axis'][0] != 3:
        raise NotImplementedError('Only axis=3 is supported')
    if layer.config['center'] != True:
        raise NotImplementedError('Only center=True is supported')
    if layer.config['scale'] != True:
        raise NotImplementedError('Only scale=True is supported')
    
    gamma = layer.weights[0]
    beta = layer.weights[1]
    moving_mean = layer.weights[2]
    moving_var = layer.weights[3]
    epsilon = layer.config['epsilon']

    a = gamma/(moving_var+epsilon)**.5
    b = beta-gamma*moving_mean/(moving_var+epsilon)**.5
    
    return [Component(layer.name, templates['BatchNormalization2D'], [
        Signal('in', layer.input),
        Signal('a', a.shape, a),
        Signal('b', b.shape, b),
        ],[Signal('out', layer.output)],{
        'nRows': layer.input[0],
        'nCols': layer.input[1],
        'nChannels': layer.input[2],
        })]

def transpile_Conv2D(layer: Layer) -> typing.List[Component]:
    if layer.config['data_format'] != 'channels_last':
        raise NotImplementedError('Only data_format="channels_last" is supported')
    if layer.config['padding'] != 'valid':
        raise NotImplementedError('Only padding="valid" is supported')
    if layer.config['strides'][0] != layer.config['strides'][1]:
        raise NotImplementedError('Only strides[0] == strides[1] is supported')
    if layer.config['kernel_size'][0] != layer.config['kernel_size'][1]:
        raise NotImplementedError('Only kernel_size[0] == kernel_size[1] is supported')
    if layer.config['dilation_rate'][0] != 1:
        raise NotImplementedError('Only dilation_rate[0] == 1 is supported')
    if layer.config['dilation_rate'][1] != 1:
        raise NotImplementedError('Only dilation_rate[1] == 1 is supported')
    if layer.config['groups'] != 1:
        raise NotImplementedError('Only groups == 1 is supported')
    if layer.config['activation'] not in ['linear', 'relu']:
        raise NotImplementedError(f'Activation {layer.config["activation"]} is not supported')
    
    if layer.config['use_bias'] == False:
        layer.weights.append(np.zeros(layer.weights[0].shape[-1]))

    conv = Component(layer.name, templates['Conv2D'], [
        Signal('in', layer.input),
        Signal('weights', layer.weights[0].shape, layer.weights[0]),
        Signal('bias', layer.weights[1].shape, layer.weights[1]),
        ],[Signal('out', layer.output)],{
        'nRows': layer.input[0],
        'nCols': layer.input[1],
        'nChannels': layer.input[2],
        'nFilters': layer.config['filters'],
        'kernelSize': layer.config['kernel_size'][0],
        'strides': layer.config['strides'][0],
        })
    
    if layer.config['activation'] == 'relu':
        activation = Component(layer.name+'_re_lu', templates['ReLU'], [Signal('in', layer.output)], [Signal('out', layer.output)])
        return [conv, activation]
    
    return [conv]

def transpile_Dense(layer: Layer, last: bool = False) -> typing.List[Component]:
    if not last and layer.config['activation'] == 'softmax':
        raise NotImplementedError('Softmax is only supported as last layer')
    if layer.config['activation'] not in ['linear', 'relu', 'softmax']:
        raise NotImplementedError(f'Activation {layer.config["activation"]} is not supported')
    if layer.config['use_bias'] == False:
        layer.weights.append(np.zeros(layer.weights[0].shape[-1]))
    
    dense = Component(layer.name, templates['Dense'], [
        Signal('in', layer.input),
        Signal('weights', layer.weights[0].shape, layer.weights[0]),
        Signal('bias', layer.weights[1].shape, layer.weights[1]),
        ],[Signal('out', layer.output)],{
        'nInputs': layer.input[0],
        'nOutputs': layer.output[0],
        })
    
    if layer.config['activation'] == 'relu':
        activation = Component(layer.name+'_re_lu', templates['ReLU'], [Signal('in', layer.output)], [Signal('out', layer.output)])
        return [dense, activation]
    
    if layer.config['activation'] == 'softmax':
        activation = Component(layer.name+'_softmax', templates['ArgMax'], [Signal('in', layer.output)], [Signal('out', (1,))], {'n': layer.output[0]})
        return [dense, activation]
    
    return [dense]

def transpile_Flatten2D(layer: Layer) -> typing.List[Component]:
    if layer.input.__len__() != 3:
        raise NotImplementedError('Only 2D inputs are supported')
    
    return [Component(layer.name, templates['Flatten2D'], [
        Signal('in', layer.input),
        ],[Signal('out', layer.output)],{
        'nRows': layer.input[0],
        'nCols': layer.input[1],
        'nChannels': layer.input[2],
        })]

def transpile_GlobalAveragePooling2D(layer: Layer) -> typing.List[Component]:
    if layer.config['data_format'] != 'channels_last':
        raise NotImplementedError('Only data_format="channels_last" is supported')
    if layer.config['keepdims']:
        raise NotImplementedError('Only keepdims=False is supported')

    return [Component(layer.name, templates['GlobalAveragePooling2D'], [
        Signal('in', layer.input),
        ],[Signal('out', layer.output)],{
        'nRows': layer.input[0],
        'nCols': layer.input[1],
        'nChannels': layer.input[2],
        'scaledInv': 1/(layer.input[0]*layer.input[1]),
        })]

def transpile_GlobalMaxPooling2D(layer: Layer) -> typing.List[Component]:
    if layer.config['data_format'] != 'channels_last':
        raise NotImplementedError('Only data_format="channels_last" is supported')
    if layer.config['keepdims']:
        raise NotImplementedError('Only keepdims=False is supported')

    return [Component(layer.name, templates['GlobalMaxPooling2D'], [
        Signal('in', layer.input),
        ],[Signal('out', layer.output)],{
        'nRows': layer.input[0],
        'nCols': layer.input[1],
        'nChannels': layer.input[2],
        })]

def transpile_Poly(layer: Layer) -> typing.List[Component]:
    return [Component(layer.name, templates['Poly'], [Signal('in', layer.input)], [Signal('out', layer.output)])]

def transpile_MaxPooling2D(layer: Layer) -> typing.List[Component]:
    if layer.config['data_format'] != 'channels_last':
        raise NotImplementedError('Only data_format="channels_last" is supported')
    if layer.config['padding'] != 'valid':
        raise NotImplementedError('Only padding="valid" is supported')
    if layer.config['pool_size'][0] != layer.config['pool_size'][1]:
        raise NotImplementedError('Only pool_size[0] == pool_size[1] is supported')
    if layer.config['strides'][0] != layer.config['strides'][1]:
        raise NotImplementedError('Only strides[0] == strides[1] is supported')
    
    return [Component(layer.name, templates['MaxPooling2D'], [Signal('in', layer.input)], [Signal('out', layer.output)],{
        'nRows': layer.input[0],
        'nCols': layer.input[1],
        'nChannels': layer.input[2],
        'poolSize': layer.config['pool_size'][0],
        'strides': layer.config['strides'][0],
        })]
