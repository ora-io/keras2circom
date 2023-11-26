const chai = require('chai');
const fs = require('fs');

const wasm_tester = require('circom_tester').wasm;

const F1Field = require('ffjavascript').F1Field;
const Scalar = require('ffjavascript').Scalar;
exports.p = Scalar.fromString('21888242871839275222246405745257275088548364400416034343698204186575808495617');
const Fr = new F1Field(exports.p);

const assert = chai.assert;

const exec = require('await-exec');

const input = require('../test/X_test/0.json');

describe('keras2circom test', function () {
    this.timeout(100000000);

    describe('softmax output', async () => {
        it('softmax output test', async () => {
            await exec('python main.py models/model.h5 && python output/circuit.py output/circuit.json test/X_test/0.json');

            const model = JSON.parse(fs.readFileSync('./output/circuit.json'));
            const output = JSON.parse(fs.readFileSync('./output/output.json'));

            const INPUT = {...model, ...input, ...output};

            const circuit = await wasm_tester('./output/circuit.circom');
            const witness = await circuit.calculateWitness(INPUT, true);
            assert(Fr.eq(Fr.e(witness[0]),Fr.e(1)));
            assert(Fr.eq(Fr.e(witness[1]),Fr.e(7)));
        });
    });

    describe('raw output', async () => {
        it('raw output test', async () => {
            await exec('python main.py models/model.h5 --raw && python output/circuit.py output/circuit.json test/X_test/0.json');

            const model = JSON.parse(fs.readFileSync('./output/circuit.json'));
            const output = JSON.parse(fs.readFileSync('./output/output.json'));

            const INPUT = {...model, ...input, ...output};

            const circuit = await wasm_tester('./output/circuit.circom');
            const witness = await circuit.calculateWitness(INPUT, true);
            assert(Fr.eq(Fr.e(witness[0]),Fr.e(1)));
        });
    });
});