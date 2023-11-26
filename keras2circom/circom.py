# Ref: https://github.com/zk-ml/uchikoma/blob/main/python/uchikoma/circom.py
from __future__ import annotations

import typing

import os
from os import path
import json
from dataclasses import dataclass

import re
import numpy as np

class SafeDict(dict):
    def __missing__(self, key):
        return '{' + key + '}'

# template string for circom
circom_template_string = '''pragma circom 2.0.0;

{include}
template Model() {brace_left}
{signal}
{component}
{main}
{brace_right}

component main = Model();
'''

templates: typing.Dict[str, Template] = {

}

def parse_shape(shape: typing.List[int]) -> str:
    '''parse shape to integers enclosed by []'''
    shape_str = ''
    for dim in shape:
        shape_str += '[{}]'.format(dim)
    return shape_str
    
def parse_index(shape: typing.List[int]) -> str:
    '''parse shape to indices enclosed by []'''
    index_str = ''
    for i in range(len(shape)):
        index_str += '[i{}]'.format(i)
    return index_str

@dataclass
class Template:
    op_name: str
    fpath: str

    args: typing.Dict[str]

    input_names: typing.List[str]   = None
    input_dims: typing.List[int]    = None
    output_names: typing.List[str]  = None
    output_dims: typing.List[int]   = None

    def __str__(self) -> str:
        args_str = ', '.join(self.args)
        args_str = '(' + args_str + ')'
        return '{:>20}{:30} {}{}{}{} \t<-- {}'.format(
            self.op_name, args_str,
            self.input_names, self.input_dims,
            self.output_names, self.output_dims,
            self.fpath)

def file_parse(fpath):
    '''parse circom file and register templates'''
    with open(fpath, 'r') as f:
        lines = f.read().split('\n')

    lines = [l for l in lines if not l.strip().startswith('//')]
    lines = ' '.join(lines)

    lines = re.sub('/\*.*?\*/', 'IGN', lines)

    funcs = re.findall('template (\w+) ?\((.*?)\) ?\{(.*?)\}', lines)
    for func in funcs:
        op_name = func[0].strip()
        args = func[1].split(',')
        main = func[2].strip()
        assert op_name not in templates, \
            'duplicated template: {} in {} vs. {}'.format(
                    op_name, templates[op_name].fpath, fpath)

        signals = re.findall('signal (\w+) (\w+)(.*?);', main)
        infos = [[] for i in range(4)]
        for sig in signals:
            sig_types = ['input', 'output']
            assert sig[0] in sig_types, sig[1] + ' | ' + main
            idx = sig_types.index(sig[0])
            infos[idx*2+0].append(sig[1])

            sig_dim = sig[2].count('[')
            infos[idx*2+1].append(sig_dim)
        templates[op_name] = Template(
                op_name, fpath,
                [a.strip() for a in args],
                *infos)


def dir_parse(dir_path, skips=[]):
    '''parse circom files in a directory'''
    names = os.listdir(dir_path)
    for name in names:
        if name in skips:
            continue

        fpath = path.join(dir_path, name)
        if os.path.isdir(fpath):
            dir_parse(fpath)
        elif os.path.isfile(fpath):
            if fpath.endswith('.circom'):
                file_parse(fpath)

@dataclass
class Signal:
    name: str
    shape: typing.List[int]
    value: typing.Any = None

    def inject_signal(self, comp_name: str) -> str:
        '''inject signal into the beginning of the circuit'''
        if self.value is not None or self.name == 'out' or self.name == 'remainder':
            return 'signal input {}_{}{};\n'.format(
                    comp_name, self.name, parse_shape(self.shape))
        return ''
    
    def inject_main(self, comp_name: str, prev_comp_name: str = None, prev_signal: Signal = None) -> str:
        '''inject signal into main'''
        inject_str = ''
        if self.value is not None or self.name == 'out' or self.name == 'remainder':
            if comp_name.endswith('softmax') and self.name == 'out':
                inject_str += '{}.out <== {}_out[0];\n'.format(
                            comp_name, comp_name)
                return inject_str
            for i in range(len(self.shape)):
                inject_str += '{}for (var i{} = 0; i{} < {}; i{}++) {{\n'.format(
                            ' '*i*4, i, i, self.shape[i], i)
            if 'activation' in comp_name or 're_lu' in comp_name:
                inject_str += '{}{}{}.{} <== {}_{}{};\n'.format(' '*(i+1)*4,
                            comp_name, parse_index(self.shape), self.name,
                            comp_name, self.name, parse_index(self.shape))
            else:
                inject_str += '{}{}.{}{} <== {}_{}{};\n'.format(' '*(i+1)*4,
                            comp_name, self.name, parse_index(self.shape),
                            comp_name, self.name, parse_index(self.shape))
            inject_str += '}'*len(self.shape)+'\n'
            return inject_str
        
        if self.shape != prev_signal.shape:
            raise ValueError('shape mismatch: {} vs. {}'.format(self.shape, prev_signal.shape))
            
        for i in range(len(self.shape)):
            inject_str += '{}for (var i{} = 0; i{} < {}; i{}++) {{\n'.format(
                            ' '*i*4, i, i, self.shape[i], i)
        
        if 'activation' in comp_name or 're_lu' in comp_name:
            inject_str += '{}{}{}.{} <== {}.{}{};\n'.format(' '*(i+1)*4,
                        comp_name, parse_index(self.shape), self.name,
                        prev_comp_name, prev_signal.name, parse_index(self.shape))
        elif 'activation' in prev_comp_name or 're_lu' in prev_comp_name:
            inject_str += '{}{}.{}{} <== {}{}.{};\n'.format(' '*(i+1)*4,
                        comp_name, self.name, parse_index(self.shape),
                        prev_comp_name, parse_index(self.shape), prev_signal.name)
        else:
            inject_str += '{}{}.{}{} <== {}.{}{};\n'.format(' '*(i+1)*4,
                        comp_name, self.name, parse_index(self.shape),
                        prev_comp_name, prev_signal.name, parse_index(self.shape))
        inject_str += '}'*len(self.shape)+'\n'
        return inject_str
    
    def inject_input_signal(self) -> str:
        '''inject the circuit input signal'''
        if self.value is not None:
            raise ValueError('input signal should not have value')
        return 'signal input in{};\n'.format(parse_shape(self.shape))
    
    def inject_output_signal(self) -> str:
        '''inject the circuit output signal'''
        if self.value is not None:
            raise ValueError('output signal should not have value')
        return 'signal output out{};\n'.format(parse_shape(self.shape)) 
    
    def inject_input_main(self, comp_name: str) -> str:
        '''inject the circuit input signal into main'''
        if self.value is not None:
            raise ValueError('input signal should not have value')
        inject_str = ''
        for i in range(len(self.shape)):
            inject_str += '{}for (var i{} = 0; i{} < {}; i{}++) {{\n'.format(
                        ' '*i*4, i, i, self.shape[i], i)
        inject_str += '{}{}.{}{} <== in{};\n'.format(' '*(i+1)*4,
                    comp_name, self.name, parse_index(self.shape),
                    parse_index(self.shape))
        inject_str += '}'*len(self.shape)+'\n'
        return inject_str
    
    def inject_output_main(self, prev_comp_name: str, prev_signal: Signal) -> str:
        '''inject the circuit output signal into main'''        
        if self.value is not None:
            raise ValueError('output signal should not have value')
        if self.shape != prev_signal.shape:
            raise ValueError('shape mismatch: {} vs. {}'.format(self.shape, prev_signal.shape))
        
        if 'softmax' in prev_comp_name:
            return 'out[0] <== {}.out;\n'.format(prev_comp_name)
        
        inject_str = ''

        for i in range(len(self.shape)):
            inject_str += '{}for (var i{} = 0; i{} < {}; i{}++) {{\n'.format(
                        ' '*i*4, i, i, self.shape[i], i)
        
        if 're_lu' in prev_comp_name:
            inject_str += '{}out{} <== {}{}.{};\n'.format(' '*(i+1)*4,
                        parse_index(self.shape),
                        prev_comp_name, parse_index(self.shape), prev_signal.name)
        else:
            inject_str += '{}out{} <== {}.{}{};\n'.format(' '*(i+1)*4,
                        parse_index(self.shape),
                        prev_comp_name, prev_signal.name, parse_index(self.shape))
        inject_str += '}'*len(self.shape)+'\n'
        return inject_str

@dataclass
class Component:
    name: str
    template: Template
    inputs: typing.List[Signal]
    outputs: typing.List[Signal]
    # optional args
    args: typing.Dict[str, typing.Any] = None

    def inject_include(self) -> str:
        '''include the component template'''
        return 'include "../{}";\n'.format(self.template.fpath)
    
    def inject_signal(self, prev_comp: Component = None, last_comp: bool = False) -> str:
        '''inject the component signals'''
        inject_str = ''
        for signal in self.inputs:
            if signal.name == 'out' or signal.name == 'remainder':
                inject_str += signal.inject_signal(self.name)
                if last_comp is True and signal.name == 'out':
                    inject_str += signal.inject_output_signal()
            elif signal.value is None and prev_comp is None:
                inject_str += signal.inject_input_signal()
            elif signal.value is not None:
                inject_str += signal.inject_signal(self.name)
        return inject_str
    
    def inject_component(self) -> str:
        '''inject the component declaration'''
        if self.template.op_name == 'ReLU':
            for signal in self.inputs:
                if signal.name == 'out':
                    output_signal = signal
                    break
            inject_str = 'component {}{};\n'.format(self.name, parse_shape(output_signal.shape))
            for i in range(len(output_signal.shape)):
                inject_str += '{}for (var i{} = 0; i{} < {}; i{}++) {{\n'.format(
                            ' '*i*4, i, i, output_signal.shape[i], i)
            inject_str += '{}{}{} = ReLU();\n'.format(' '*(i+1)*4,
                        self.name, parse_index(output_signal.shape))
            inject_str += '}'*len(output_signal.shape)+'\n'
            return inject_str

        return 'component {} = {}({});\n'.format(
            self.name, self.template.op_name, self.parse_args(self.template.args, self.args))
    
    def inject_main(self, prev_comp: Component = None, last_comp: bool = False) -> str:
        '''inject the component main'''
        inject_str = ''
        for signal in self.inputs:
            if signal.value is not None or signal.name == 'out' or signal.name == 'remainder':
                inject_str += signal.inject_main(self.name)
            elif prev_comp is None:
                inject_str += signal.inject_input_main(self.name)
            else:
                for sig in prev_comp.inputs:
                    if sig.name == 'out':
                        output_signal = sig
                        break
                if output_signal is None:
                    output_signal = prev_comp.outputs[0]
                inject_str += signal.inject_main(self.name, prev_comp.name, output_signal)
                print
        if last_comp:
            for signal in self.inputs:
                if signal.name == 'out':
                    inject_str += signal.inject_output_main(self.name, signal)
                    break
            for signal in self.outputs:
                inject_str += signal.inject_output_main(self.name, signal)
        return inject_str

    def to_json(self, dec: int) -> typing.Dict[str, typing.Any]:
        '''convert the component params to json format'''
        json_dict = {}
        for signal in self.inputs:
            if signal.value is not None:
                if signal.name == 'bias' or signal.name == 'b':
                    scaling = float(10**(2*dec))
                else:
                    scaling = float(10**dec)
                value = [str(int(v*scaling)) for v in signal.value.flatten().tolist()]
                # reshape the value to match the circom shape
                if len(signal.shape) > 1:
                    value = np.array(value).reshape(signal.shape).tolist()
                json_dict.update({f'{self.name}_{signal.name}': value})
        return json_dict
    
    @staticmethod
    def parse_args(template_args: typing.List[str], args: typing.Dict[str, typing.Any]) -> str:
        '''parse the args to a format string, ready to be injected'''
        args_str = '{'+'}, {'.join(template_args)+'}'
        return args_str.format(**args)

@dataclass
class Circuit:
    components: typing.List[Component]

    def __init__(self):
        self.components = []

    def add_component(self, component: Component):
        self.components.append(component)
    
    def add_components(self, components: typing.List[Component]):
        self.components.extend(components)

    def inject_include(self) -> str:
        '''inject the include statements'''
        inject_str = []
        for component in self.components:
            inject_str.append(component.inject_include())
        return ''.join(set(inject_str))

    def inject_signal(self) -> str:
        '''inject the signal declarations'''
        inject_str = self.components[0].inject_signal()
        for i in range(1, len(self.components)):
            inject_str += self.components[i].inject_signal(self.components[i-1], i==len(self.components)-1)
        return inject_str

    def inject_component(self) -> str:
        '''inject the component declarations'''
        inject_str = ''
        for component in self.components:
            inject_str += component.inject_component()
        return inject_str
    
    def inject_main(self) -> str:
        '''inject the main template'''
        inject_str = self.components[0].inject_main()
        for i in range(1, len(self.components)):
            inject_str += self.components[i].inject_main(self.components[i-1], i==len(self.components)-1)
        return inject_str

    def to_circom(self) -> str:
        '''convert the circuit to a circom file'''
        return circom_template_string.format(**{
            'include': self.inject_include(),
            'brace_left': '{',
            'signal': self.inject_signal(),
            'component': self.inject_component(),
            'main': self.inject_main(),
            'brace_right': '}',
        })

    def to_json(self, dec: int) -> str:
        '''convert the model weights to json format'''
        json_dict = {}

        for component in self.components:
            json_dict.update(component.to_json(dec))
        
        return json.dumps(json_dict)