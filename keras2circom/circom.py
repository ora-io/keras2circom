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

circom_template_string = '''
pragma circom 2.0.0;

{include}
template Model() {brace_left}
{signal}
{main}
{brace_right}

component main = Model();
'''

templates: typing.Dict[str, Template] = {

}

def inject(template, key, code) -> str:
    ''' helper function to auto inject sections. '''
    # append key to the end
    code += '{{{}}}'.format(key)
    return template.format_map(SafeDict({key: code}))

def inject_main(template, code) -> str:
    return inject(template, 'main', code + '\n')

def inject_signal(template, code) -> str:
    return inject(template, 'signal', code + '\n')

def inject_include(template, code) -> str:
    return inject(template, 'include', code + '\n')

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

@dataclass
class Component:
    name: str
    template: Template
    inputs: typing.List[Signal]
    outputs: typing.List[Signal]
    # optional args
    args: typing.Dict[str, typing.Any] = None

@dataclass
class Circuit:
    components: typing.List[Component]

    def __init__(self):
        self.components = []

    def add_component(self, component: Component):
        self.components.append(component)
    
    def add_components(self, components: typing.List[Component]):
        self.components.extend(components)

    def to_circom(self):
        template = circom_template_string
        for template in templates.values():
            template.inject_include(template)

        for component in self.components:
            template.inject_signal(template, component.inject_signal())
            template.inject_main(template, component.inject_main())

        return template