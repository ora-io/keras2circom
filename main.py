""" Transpile a Keras model to a CIRCOM circuit.

Usage:
    keras2circom.py <model.h5> [-o <output>] [--raw]
    keras2circom.py (-h | --help)

Options:
    -h --help                       Show this screen.
    -o <output> --output=<output>   Output file [default: model.circom].
    --raw                           Output raw model outputs instead of the argmax of outputs [default: False].

"""
from docopt import docopt

from keras2circom import transpiler

def main():
    """ Main entry point of the app """
    args = docopt(__doc__)
    print(args)

if __name__ == "__main__":
    """ This is executed when run from the command line """
    main()