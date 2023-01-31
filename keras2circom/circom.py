# Ref: https://github.com/zk-ml/uchikoma/blob/main/python/uchikoma/circom.py

from __future__ import annotations

import typing

import os
from os import path
import json
import numpy as np
from dataclasses import dataclass

import re

class SafeDict(dict):
    def __missing__(self, key):
        return '{' + key + '}'

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

def parse_shape(shape: typing.List[int]):
    shape_str = ''
    for dim in shape:
        shape_str += '[{}]'.format(dim)
    return shape_str
    
def parse_index(shape: typing.List[int]):
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

    def __str__(self):
        args_str = ', '.join(self.args)
        args_str = '(' + args_str + ')'
        return '{:>20}{:30} {}{}{}{} \t<-- {}'.format(
            self.op_name, args_str,
            self.input_names, self.input_dims,
            self.output_names, self.output_dims,
            self.fpath)

def file_parse(fpath):
    #  print('register circom file: ', fpath)
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

# TODO: review Signal, Component, Circuit
@dataclass
class Signal:
    name: str
    shape: typing.List[int]
    value: typing.Any = None

    def inject_signal(self, compName: str):
        if self.value is not None:
            return 'signal input {}_{}{};\n'.format(
                    compName, self.name, parse_shape(self.shape))
        return ''
    
    def inject_main(self, compName: str, prevSignal: Signal):
        inject_str = ''
        if self.value is None:
            if self.shape != prevSignal.shape:
                raise ValueError('shape mismatch: {} vs. {}'.format(self.shape, prevSignal.shape))
            for i in range(len(self.shape)):
                inject_str += '{}for (var i{} = 0; i{} < {}; i{}++) {{\n'.format(
                            ' '*i*4, i, i, self.shape[i], i)
            inject_str += '{}{}_{}{} <== {}_{}{};\n'.format(' '*(i+1)*4,
                        compName, self.name, parse_index(self.shape),
                        compName, prevSignal.name, parse_index(self.shape))
        return inject_str
    
    def inject_input_signal(self):
        if self.value is not None:
            raise ValueError('input signal should not have value')
        return 'signal input in{};\n'.format(parse_shape(self.shape))
    
    def inject_output_signal(self):
        if self.value is not None:
            raise ValueError('output signal should not have value')
        return 'signal output out{};\n'.format(parse_shape(self.shape))
    
    # TODO: inject_output_main
    


@dataclass
class Component:
    name: str
    template: Template
    inputs: typing.List[Signal]
    outputs: typing.List[Signal]
    # optional args
    args: typing.Dict[str, typing.Any] = None

    def inject_include(self):
        return 'include "{}";\n'.format(self.template.fpath)
    
    def inject_signal(self, prevComponent: Component = None, lastComponent: bool = False):
        inject_str = ''
        for signal in self.inputs:
            if signal.value is None and prevComponent is None:
                inject_str += signal.inject_input_signal()
            elif signal.value is not None:
                inject_str += signal.inject_signal(self.name)
        for signal in self.outputs:
            if signal.value is None and lastComponent is True:
                inject_str += signal.inject_output_signal()
        return inject_str
    
    def inject_component(self):
        if self.template.op_name == 'ReLU':
            inject_str = 'component {}{};\n'.format(self.name, parse_shape(self.outputs[0].shape))
            for i in range(len(self.outputs[0].shape)):
                inject_str += '{}for (var i{} = 0; i{} < {}; i{}++) {{\n'.format(
                            ' '*i*4, i, i, self.outputs[0].shape[i], i)
            inject_str += '{}{}{} <== ReLU();\n'.format(' '*(i+1)*4,
                        self.name, parse_index(self.outputs[0].shape))
            return inject_str
        
        # TODO: need to handle scaling with Poly
        if self.template.op_name == 'Poly':
            return ''

        return 'component {} = {}({});\n'.format(
            self.name, self.template.op_name, self.parse_args(self.template.args, self.args))

    # TODO: fix inject_main
    def inject_main(self, prevComponent: Component, lastComponent: bool = False):
        inject_str = ''
        for signal in self.inputs:
            if signal.value is None:
                inject_str += signal.inject_main(self.name, prevComponent.outputs[0])
        if lastComponent:
            for signal in self.outputs:
                if signal.value is None:
                    inject_str += signal.inject_main(self.name, prevComponent.outputs[0])
        return inject_str
    
    @staticmethod
    def parse_args(template_args: typing.List[str], args: typing.Dict[str, typing.Any]):
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

    def inject_include(self):
        inject_str = []
        for component in self.components:
            inject_str.append(component.inject_include())
        return ''.join(set(inject_str))

    def inject_signal(self):
        inject_str = self.components[0].inject_signal()
        for i in range(1, len(self.components)):
            inject_str += self.components[i].inject_signal(self.components[i-1], i==len(self.components)-1)
        return inject_str

    def inject_component(self):
        inject_str = ''
        for component in self.components:
            inject_str += component.inject_component()
        return inject_str
    
    # TODO: fix inject_main
    def inject_main(self):
        return ''
        

    def to_circom(self):
        return circom_template_string.format(**{
            'include': self.inject_include(),
            'brace_left': '{',
            'signal': self.inject_signal(),
            'component': self.inject_component(),
            'main': '',
            'brace_right': '}',
        })