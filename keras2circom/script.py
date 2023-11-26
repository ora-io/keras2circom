from .circom import Circuit, Component

# template string for circuit.py
python_template_string = '''""" Make an interger-only circuit of the corresponding CIRCOM circuit.

Usage:
    circuit.py <circuit.json> <input.json> [-o <output>]
    circuit.py (-h | --help)

Options:
    -h --help                               Show this screen.
    -o <output> --output=<output>           Output directory [default: output].

"""

from docopt import docopt
import json

try:
    from keras2circom.util import *
except:
    import sys
    import os
    # add parent directory to sys.path
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from keras2circom.util import *

def inference(input, circuit):
    out = input['in']
    output = {brackets}
    
{components}
    return out, output


def main():
    """ Main entry point of the app """
    args = docopt(__doc__)
    
    # parse input.json
    with open(args['<input.json>']) as f:
        input = json.load(f)
    
    # parse circuit.json
    with open(args['<circuit.json>']) as f:
        circuit = json.load(f)

    out, output = inference(input, circuit)

    # write output.json
    with open(args['--output'] + '/output.json', 'w') as f:
        json.dump(output, f)

if __name__ == "__main__":
    """ This is executed when run from the command line """
    main()
'''


def to_py(circuit: Circuit, dec: int) -> str:
    comp_str = ""

    for component in circuit.components:
        comp_str += transpile_component(component, dec)

    return python_template_string.format(
        brackets="{}",
        components=comp_str,
    )

def transpile_component(component: Component, dec: int) -> str:
    comp_str = ""
    if component.template.op_name == "AveragePooling2D":
        comp_str += "    out, remainder = AveragePooling2DInt({nRows}, {nCols}, {nChannels}, {poolSize}, {strides}, {input})\n".format(
            nRows=component.args["nRows"],
            nCols=component.args["nCols"],
            nChannels=component.args["nChannels"],
            poolSize=component.args["poolSize"],
            strides=component.args["strides"],
            input="out"
        )
        comp_str += "    output['{name}_out'] = out\n".format(
            name=component.name,
        )
        comp_str += "    output['{name}_remainder'] = remainder\n".format(
            name=component.name,
        )
        return comp_str+"\n"
    
    elif component.template.op_name == "BatchNormalization2D":
        comp_str += "    out, remainder = BatchNormalizationInt({nRows}, {nCols}, {nChannels}, {n}, {input}, {a}, {b})\n".format(
            nRows=component.args["nRows"],
            nCols=component.args["nCols"],
            nChannels=component.args["nChannels"],
            n=component.args["n"],
            input="out",
            a="circuit['{name}_a']".format(name=component.name),
            b="circuit['{name}_b']".format(name=component.name),
        )
        comp_str += "    output['{name}_out'] = out\n".format(
            name=component.name,
        )
        comp_str += "    output['{name}_remainder'] = remainder\n".format(
            name=component.name,
        )
        return comp_str+"\n"
    
    elif component.template.op_name == "Conv1D":
        comp_str += "    out, remainder = Conv1DInt({nInputs}, {nChannels}, {nFilters}, {kernelSize}, {strides}, {n}, {input}, {weights}, {bias})\n".format(
            nInputs=component.args["nInputs"],
            nChannels=component.args["nChannels"],
            nFilters=component.args["nFilters"],
            kernelSize=component.args["kernelSize"],
            strides=component.args["strides"],
            n=component.args["n"],
            input="out",
            weights="circuit['{name}_weights']".format(name=component.name),
            bias="circuit['{name}_bias']".format(name=component.name),
        )
        comp_str += "    output['{name}_out'] = out\n".format(
            name=component.name,
        )
        comp_str += "    output['{name}_remainder'] = remainder\n".format(
            name=component.name,
        )
        return comp_str+"\n"
    
    elif component.template.op_name == "Conv2D":
        comp_str += "    out, remainder = Conv2DInt({nRows}, {nCols}, {nChannels}, {nFilters}, {kernelSize}, {strides}, {n}, {input}, {weights}, {bias})\n".format(
            nRows=component.args["nRows"],
            nCols=component.args["nCols"],
            nChannels=component.args["nChannels"],
            nFilters=component.args["nFilters"],
            kernelSize=component.args["kernelSize"],
            strides=component.args["strides"],
            n=component.args["n"],
            input="out",
            weights="circuit['{name}_weights']".format(name=component.name),
            bias="circuit['{name}_bias']".format(name=component.name),
        )
        comp_str += "    output['{name}_out'] = out\n".format(
            name=component.name,
        )
        comp_str += "    output['{name}_remainder'] = remainder\n".format(
            name=component.name,
        )
        return comp_str+"\n"
    
    elif component.template.op_name == "Dense":
        comp_str += "    out, remainder = DenseInt({nInputs}, {nOutputs}, {n}, {input}, {weights}, {bias})\n".format(
            nInputs=component.args["nInputs"],
            nOutputs=component.args["nOutputs"],
            n=component.args["n"],
            input="out",
            weights="circuit['{name}_weights']".format(name=component.name),
            bias="circuit['{name}_bias']".format(name=component.name),
        )
        comp_str += "    output['{name}_out'] = out\n".format(
            name=component.name,
        )
        comp_str += "    output['{name}_remainder'] = remainder\n".format(
            name=component.name,
        )
        return comp_str+"\n"
    
    elif component.template.op_name == "GlobalAveragePooling2D":
        comp_str += "    out, remainder = GlobalAveragePooling2DInt({nRows}, {nCols}, {nChannels}, {input})\n".format(
            nRows=component.args["nRows"],
            nCols=component.args["nCols"],
            nChannels=component.args["nChannels"],
            input="out"
        )
        comp_str += "    output['{name}_out'] = out\n".format(
            name=component.name,
        )
        comp_str += "    output['{name}_remainder'] = remainder\n".format(
            name=component.name,
        )
        return comp_str+"\n"
    
    elif component.template.op_name == "GlobalMaxPooling2D":
        comp_str += "    out = GlobalMaxPooling2DInt({nRows}, {nCols}, {nChannels}, {input})\n".format(
            nRows=component.args["nRows"],
            nCols=component.args["nCols"],
            nChannels=component.args["nChannels"],
            input="out"
        )
        comp_str += "    output['{name}_out'] = out\n".format(
            name=component.name,
        )
        return comp_str+"\n"
    
    elif component.template.op_name == "MaxPooling2D":
        comp_str += "    out = MaxPooling2DInt({nRows}, {nCols}, {nChannels}, {poolSize}, {strides}, {input})\n".format(
            nRows=component.args["nRows"],
            nCols=component.args["nCols"],
            nChannels=component.args["nChannels"],
            poolSize=component.args["poolSize"],
            strides=component.args["strides"],
            input="out"
        )
        comp_str += "    output['{name}_out'] = out\n".format(
            name=component.name,
        )
        return comp_str+"\n"
    
    elif component.template.op_name == "Flatten2D":
        comp_str += "    out = Flatten2DInt({nRows}, {nCols}, {nChannels}, {input})\n".format(
            nRows=component.args["nRows"],
            nCols=component.args["nCols"],
            nChannels=component.args["nChannels"],
            input="out"
        )
        comp_str += "    output['{name}_out'] = out\n".format(
            name=component.name,
        )
        return comp_str+"\n"
    
    elif component.template.op_name == "ReLU":
        nRows, nCols, nChannels = component.inputs[0].shape
        comp_str += "    out = ReLUInt({nRows}, {nCols}, {nChannels}, {input})\n".format(
            nRows=nRows,
            nCols=nCols,
            nChannels=nChannels,
            input="out"
        )
        comp_str += "    output['{name}_out'] = out\n".format(
            name=component.name,
        )
        return comp_str+"\n"
    
    elif component.template.op_name == "ArgMax":
        comp_str += "    out = ArgMaxInt(out)\n"
        comp_str += "    output['{name}_out'] = out\n".format(
            name=component.name,
        )
        return comp_str+"\n"
    
    else:
        raise ValueError("Unknown component type: {}".format(component.template.op_name))