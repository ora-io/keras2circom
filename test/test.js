const chai = require('chai');
const fs = require('fs');

const wasm_tester = require('circom_tester').wasm;

const F1Field = require('ffjavascript').F1Field;
const Scalar = require('ffjavascript').Scalar;
exports.p = Scalar.fromString('21888242871839275222246405745257275088548364400416034343698204186575808495617');
const Fr = new F1Field(exports.p);

const assert = chai.assert;

const exec = require('await-exec');

const best_practice = require('../models/best_practice.json');
const alt_model = require('../models/alt_model.json');

function softmax(arr) {
    return arr.map(function(value,index) { 
      return Math.exp(value) / arr.map( function(y /*value*/){ return Math.exp(y) } ).reduce( function(a,b){ return a+b })
    })
}

describe('keras2circom test', function () {
    this.timeout(100000000);

    describe('models/best_practice.h5', async () => {
        it('softmax output', async () => {
            await exec('python main.py models/best_practice.h5 -o best_practice');

            const json = JSON.parse(fs.readFileSync('./best_practice/circuit.json'));

            let INPUT = {};
            
            for (const [key, value] of Object.entries(json)) {
                if (Array.isArray(value)) {
                    let tmpArray = [];
                    for (let i = 0; i < value.flat().length; i++) {
                        tmpArray.push(Fr.e(value.flat()[i]));
                    }
                    INPUT[key] = tmpArray;
                } else {
                    INPUT[key] = Fr.e(value);
                }
            }
            let tmpArray = [];
            for (let i=0; i < best_practice['X'].length; i++) {
                tmpArray.push(Fr.e(best_practice['X'][i]));
            }
            INPUT['in'] = tmpArray;

            const circuit = await wasm_tester('./best_practice/circuit.circom');

            const witness = await circuit.calculateWitness(INPUT, true);

            assert(Fr.eq(Fr.e(witness[0]),Fr.e(1)));
            assert(Fr.eq(Fr.e(witness[1]),Fr.e(7)));
        });

        it('raw output', async () => {
            await exec('python main.py models/best_practice.h5 -o best_practice_raw --raw');

            const json = JSON.parse(fs.readFileSync('./best_practice_raw/circuit.json'));

            let INPUT = {};
            
            for (const [key, value] of Object.entries(json)) {
                if (Array.isArray(value)) {
                    let tmpArray = [];
                    for (let i = 0; i < value.flat().length; i++) {
                        tmpArray.push(Fr.e(value.flat()[i]));
                    }
                    INPUT[key] = tmpArray;
                } else {
                    INPUT[key] = Fr.e(value);
                }
            }
            let tmpArray = [];
            for (let i=0; i < best_practice['X'].length; i++) {
                tmpArray.push(Fr.e(best_practice['X'][i]));
            }
            INPUT['in'] = tmpArray;

            const circuit = await wasm_tester('./best_practice_raw/circuit.circom');
            
            const witness = await circuit.calculateWitness(INPUT, true);
            
            assert(Fr.eq(Fr.e(witness[0]),Fr.e(1)));

            const scale = 1E-51;

            let predicted = [];
            for (var i=0; i<best_practice['y'].length; i++) {
                predicted.push(parseFloat(Fr.toString(Fr.e(witness[i+1]))) * scale);
            }

            let ape = 0;

            for (var i=0; i<best_practice['y'].length; i++) {
                const actual = best_practice['y'][i];
                console.log('actual', actual, 'predicted', predicted[i]);
                ape += Math.abs((predicted[i]-actual)/actual);
            }

            const mape = ape/best_practice['y'].length;

            console.log('mean absolute % error', mape);
        });
    });

    describe('models/alt_model.h5', async () => {
        it('softmax output', async () => {
            await exec('python main.py models/alt_model.h5 -o alt_model');

            const json = JSON.parse(fs.readFileSync('./alt_model/circuit.json'));

            let INPUT = {};
            
            for (const [key, value] of Object.entries(json)) {
                if (Array.isArray(value)) {
                    let tmpArray = [];
                    for (let i = 0; i < value.flat().length; i++) {
                        tmpArray.push(Fr.e(value.flat()[i]));
                    }
                    INPUT[key] = tmpArray;
                } else {
                    INPUT[key] = Fr.e(value);
                }
            }
            let tmpArray = [];
            for (let i=0; i < alt_model['X'].length; i++) {
                tmpArray.push(Fr.e(alt_model['X'][i]));
            }
            INPUT['in'] = tmpArray;

            const circuit = await wasm_tester('./alt_model/circuit.circom');

            const witness = await circuit.calculateWitness(INPUT, true);

            assert(Fr.eq(Fr.e(witness[0]),Fr.e(1)));
            assert(Fr.eq(Fr.e(witness[1]),Fr.e(7)));
        });

        it('raw output', async () => {
            await exec('python main.py models/alt_model.h5 -o alt_model_raw --raw');

            const json = JSON.parse(fs.readFileSync('./alt_model_raw/circuit.json'));

            let INPUT = {};
            
            for (const [key, value] of Object.entries(json)) {
                if (Array.isArray(value)) {
                    let tmpArray = [];
                    for (let i = 0; i < value.flat().length; i++) {
                        tmpArray.push(Fr.e(value.flat()[i]));
                    }
                    INPUT[key] = tmpArray;
                } else {
                    INPUT[key] = Fr.e(value);
                }
            }
            let tmpArray = [];
            for (let i=0; i < alt_model['X'].length; i++) {
                tmpArray.push(Fr.e(alt_model['X'][i]));
            }
            INPUT['in'] = tmpArray;

            const circuit = await wasm_tester('./alt_model_raw/circuit.circom');
            
            const witness = await circuit.calculateWitness(INPUT, true);
            
            assert(Fr.eq(Fr.e(witness[0]),Fr.e(1)));

            const scale = 1E-60;

            let predicted_raw = [];
            for (var i=0; i<alt_model['y'].length; i++) {
                predicted_raw.push(parseFloat(Fr.toString(Fr.e(witness[i+1]))) * scale);
            }

            const predicted = softmax(predicted_raw);

            let ape = 0;

            for (var i=0; i<alt_model['y'].length; i++) {
                const actual = alt_model['y'][i]
                console.log('actual', actual, 'predicted', predicted[i]);
                ape += Math.abs((predicted[i]-actual)/actual);
            }

            const mape = ape/alt_model['y'].length;

            console.log('mean absolute % error', mape);
        });
    });
});