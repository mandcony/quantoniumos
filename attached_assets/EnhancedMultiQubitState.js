/**
 * EnhancedMultiQubitState: A class to represent and manipulate multi-qubit states.
 */
export class EnhancedMultiQubitState {
    /**
     * Creates a new multi-qubit state.
     * @param {number} numQubits - The number of qubits in the state.  Must be a positive integer.
     * @throws {Error} If numQubits is not a positive integer.
     */
    constructor(numQubits) {
        if (!Number.isInteger(numQubits) || numQubits <= 0) {
            throw new Error("Number of qubits must be a positive integer.");
        }

        this.numQubits = numQubits;
        this.stateTensor = this.initializeTensor();
        this.entanglementMap = {}; // Track qubit correlations (simulated)
    }

    /**
     * Initializes the state tensor with random complex amplitudes.
     * The dimension of the tensor is 2^numQubits.  The amplitudes are normalized
     * so that the sum of their squared magnitudes equals 1.
     * @returns {number[]} An array representing the state tensor, with complex amplitudes represented as { real: number, imag: number }.
     * @private
     */
    initializeTensor() {
        const dimension = 2 ** this.numQubits;
        const tensor = new Array(dimension);
        let sumOfSquares = 0;

        // Assign random complex amplitudes and track the sum of their squares.
        for (let i = 0; i < dimension; i++) {
            const real = Math.random() * 2 - 1; // Random number between -1 and 1
            const imag = Math.random() * 2 - 1; // Random number between -1 and 1
            tensor[i] = { real, imag };
            sumOfSquares += real * real + imag * imag;
        }

        // Normalize the amplitudes
        const normalizationFactor = Math.sqrt(sumOfSquares);
        for (let i = 0; i < dimension; i++) {
            tensor[i].real /= normalizationFactor;
            tensor[i].imag /= normalizationFactor;
        }

        return tensor;
    }

    /**
     * Applies a quantum gate to the state tensor.  This is a matrix-vector multiplication.
     * @param {number[][]} gateMatrix - The gate matrix to apply. Must be a square matrix with dimensions 2^n x 2^n, where n is an integer.
     * @throws {Error} If the gate matrix is invalid.
     */
    applyGate(gateMatrix) {
        if (!Array.isArray(gateMatrix)) {
            throw new Error("Gate matrix must be an array.");
        }

        const matrixDimension = gateMatrix.length;
        if ((matrixDimension & (matrixDimension - 1)) !== 0 || matrixDimension !== this.stateTensor.length) {
            throw new Error("Gate matrix must be a square matrix with dimensions that are a power of 2 and equal the state tensor length.");
        }

        //Matrix-vector product implementation, handling complex numbers
        const newStateTensor = new Array(this.stateTensor.length);

        for (let i = 0; i < this.stateTensor.length; i++) {
            let newReal = 0;
            let newImag = 0;
            for (let j = 0; j < this.stateTensor.length; j++) {
                //Complex multiplication: (a + bi) * (c + di) = (ac - bd) + (ad + bc)i
                const gateElement = gateMatrix[i][j];
                if(typeof gateElement !== 'object' || !('real' in gateElement) || !('imag' in gateElement)) {
                    throw new Error("Gate matrix element is not a complex number object {real:number, imag:number}.");
                }
                const stateElement = this.stateTensor[j];

                newReal += (gateElement.real * stateElement.real) - (gateElement.imag * stateElement.imag);
                newImag += (gateElement.real * stateElement.imag) + (gateElement.imag * stateElement.real);
            }
            newStateTensor[i] = {real: newReal, imag: newImag};
        }

        this.stateTensor = newStateTensor;
    }



    /**
     * Entangles two qubits. This is a simplified simulation; real entanglement requires specific quantum operations.
     * @param {number} qubitA - The index of the first qubit.
     * @param {number} qubitB - The index of the second qubit.
     * @throws {Error} If the qubit indices are invalid.
     */
    entangle(qubitA, qubitB) {
        if (!Number.isInteger(qubitA) || !Number.isInteger(qubitB) || qubitA < 0 || qubitA >= this.numQubits || qubitB < 0 || qubitB >= this.numQubits || qubitA === qubitB) {
            throw new Error("Invalid qubit indices for entanglement.");
        }

        const key = `${qubitA}-${qubitB}`;
        this.entanglementMap[key] = true;
        console.log(`Entangled Qubit ${qubitA} with Qubit ${qubitB}`);  // Consider removing or using a logger
    }

    /**
     * Measures the multi-qubit state and returns a bit string representing the outcome.
     * This collapses the state to a single outcome based on the probabilities.
     * @returns {string} A bit string representing the measurement outcome.
     */
    measure() {
        // Calculate probabilities (magnitude squared of complex amplitudes)
        let probabilities = this.stateTensor.map(amp => amp.real * amp.real + amp.imag * amp.imag);

        // Normalize the probabilities to ensure they sum to 1.
        const sumOfProbabilities = probabilities.reduce((sum, probability) => sum + probability, 0);
        probabilities = probabilities.map(probability => probability / sumOfProbabilities);

        let randomNumber = Math.random();
        let cumulativeProbability = 0;

        for (let i = 0; i < probabilities.length; i++) {
            cumulativeProbability += probabilities[i];
            if (randomNumber <= cumulativeProbability) {
                return i.toString(2).padStart(this.numQubits, '0');
            }
        }

        // Should not happen, but return the last state if there are floating point precision issues.
        return (probabilities.length - 1).toString(2).padStart(this.numQubits, '0');
    }
}