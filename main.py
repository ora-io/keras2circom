""" Transpile a Keras model to a CIRCOM circuit.

Usage:
    main.py <model.h5> [-o <output>] [--raw] [-d <decimals>]
    main.py (-h | --help)

Options:
    -h --help                               Show this screen.
    -o <output> --output=<output>           Output directory [default: output].
    --raw                                   Output raw model outputs instead of the argmax of outputs [default: False].
    -d <decimals> --decimals=<decimals>     Number of decimals for model precision [default: 18].

"""
from docopt import docopt

from keras2circom import circom, transpiler

def main():
    """ Main entry point of the app """
    args = docopt(__doc__)
    circom.dir_parse('node_modules/circomlib-ml/circuits/', skips=['util.circom', 'circomlib-matrix', 'circomlib', 'crypto'])
    transpiler.transpile(args['<model.h5>'], args['--output'], args['--raw'], args['--decimals'])

if __name__ == "__main__":
    """ This is executed when run from the command line """
    main()