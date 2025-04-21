import { MultiQubitState } from './multiQubitState';
import GeometricContainer from './GeometricContainer';

export default class QuantumNovaSystem {
    constructor(numQubits = 3) {
        this.numQubits = numQubits;
        this.qState = new MultiQubitState(numQubits);
        this.containers = [];
    }

    createContainer(id, vertices, transformations = [], material = {}) {
        const gc = new GeometricContainer(id, vertices, transformations, material);
        this.containers.push(gc);
        return gc;
    }
    // This function was copied from prior versions.
    runQuantumDemo() {
        const hMatrix = [
            [
                (amp) => `(1/sqrt(2))*${amp}`,
                (amp) => `(1/sqrt(2))*${amp}`
            ],
            [
                (amp) => `(1/sqrt(2))*${amp}`,
                (amp) => `(-1/sqrt(2))*${amp}`
            ],
        ];
        this.qState.applySingleQubitGate(hMatrix, 0);

        if (this.numQubits >= 2) {
            this.qState.applyTwoQubitGate('CNOT', 0, 1);
        }

        const measured = this.qState.measureAll();

        return {
            measurement: measured,
            amplitudes: this.qState.getAmplitudeInfo(),
        };
    }
    //Fixed
    runGeometryDemo(data) {
        const quantumResult = this.runQuantumDemo()
        for (let c of this.containers) {
              // Apply data to wave vertices
             c.applyDataToWaveVertices(data);
            c.applyAllTransformations(quantumResult.amplitudes);

        }
    }

    retrieveDataIfResonant(container, freq) {
        return this.vibrationalEngine.retrieveData(container, freq);
    }
}