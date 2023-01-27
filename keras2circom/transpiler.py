from .circom import *
from .model import *

def transpile(filename: str, output: str = 'output.circom', raw: bool = False) -> Circuit:
    model = Model(filename, raw)

    circuit = Circuit()
    for layer in model.layers[:-1]:
        circuit.add_components(transpile_layer(layer))
    
    if model.layers[-1].op in supported_ops:
        pass

    return circuit

def transpile_layer(layer: Layer, last: bool = False) -> typing.List[Component]:
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
        return []
    if layer.op == 'Flatten':
        return []
    if layer.op == 'GlobalAveragePooling2D':
        return []
    if layer.op == 'GlobalMaxPooling2D':
        return []
    if layer.op == 'Lambda':
        return [] # only for polynomial activation in the form of `Lambda(lambda x: x**2+x)`
    if layer.op == 'MaxPooling2D':
        return []
    
    raise NotImplementedError(f'Layer {layer.op} is not supported yet.')

# TODO: handle scaling
def transpile_ArgMax(layer: Layer) -> typing.List[Component]:
    return [Component(layer.name, 'ArgMax', [Signal('in', layer.input)], [Signal('out', (1,))], {'n': layer.input[0]})]

def transpile_ReLU(layer: Layer) -> typing.List[Component]:
    return [Component(layer.name, 'ReLU', [Signal('in', layer.input)], [Signal('out', layer.output)])]

def transpile_AveragePooling2D(layer: Layer) -> typing.List[Component]:
    if layer.config['data_format'] != 'channels_last':
        raise NotImplementedError('Only data_format="channels_last" is supported')
    if layer.config['padding'] != 'valid':
        raise NotImplementedError('Only padding="valid" is supported')
    if layer.config['pool_size'][0] != layer.config['pool_size'][1]:
        raise NotImplementedError('Only pool_size[0] == pool_size[1] is supported')
    if layer.config['strides'][0] != layer.config['strides'][1]:
        raise NotImplementedError('Only strides[0] == strides[1] is supported')
    
    return [Component(layer.name, 'AveragePooling2D', [Signal('in', layer.input)], [Signal('out', layer.output)],{
        'nRows': layer.input[0],
        'nCols': layer.input[1],
        'nChannels': layer.input[2],
        'pool_size': layer.config['pool_size'][0],
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
    
    return [Component(layer.name, 'BatchNormalization2D', [
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
        layer.weights.append(np.zeros(layer.weights[0].shape[3]))

    conv = Component(layer.name, 'Conv2D', [
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
        activation = Component(layer.name+'_relu', 'ReLU', [Signal('in', layer.output)], [Signal('out', layer.output)])
        return [conv, activation]
    
    return [conv]