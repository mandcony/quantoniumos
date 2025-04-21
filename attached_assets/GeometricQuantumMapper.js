/**
 * GeometricQuantumMapper: Maps quantum states to geometric spaces.
 *  Provides methods for encoding, and transforming state vectors using various geometric mappings.
 */
export class GeometricQuantumMapper {
    /**
     * Creates a new GeometricQuantumMapper.
     * @param {number} dimensions - The number of dimensions for the geometric space. Must be a positive integer.
     * @throws {Error} If dimensions is not a positive integer.
     */
    constructor(dimensions) {
        if (!Number.isInteger(dimensions) || dimensions <= 0) {
            throw new Error("Dimensions must be a positive integer.");
        }

        this.dimensions = dimensions;
        this.space = this.initializeGeometricSpace();
    }

    /**
     * Initializes the geometric space with random coordinates.
     * Coordinates are uniformly distributed between -1 and 1.
     * @returns {number[]} An array representing the geometric space coordinates.
     * @private
     */
    initializeGeometricSpace() {
        const space = new Array(this.dimensions);
        for (let i = 0; i < this.dimensions; i++) {
            space[i] = Math.random() * 2 - 1;
        }
        return space;
    }

    /**
     * Encodes a qubit state (string of "0"s and "1"s) into a numerical vector.
     * "1" is mapped to 1, and "0" is mapped to -1.
     * @param {string} qubitState - The qubit state to encode. Must be a string containing only "0" and "1" characters.
     * @returns {number[]} An array representing the encoded state.
     * @throws {Error} If the qubitState is invalid.
     */
    encodeState(qubitState) {
        if (typeof qubitState !== 'string' || !/^[01]+$/.test(qubitState)) {
            throw new Error("Qubit state must be a string containing only '0' and '1' characters.");
        }

        const encodedState = new Array(qubitState.length);
        for (let i = 0; i < qubitState.length; i++) {
            encodedState[i] = (qubitState[i] === "1" ? 1 : -1);
        }
        return encodedState;
    }

    /**
     * Transforms a state vector using a spherical transformation (sine function).
     * Each coordinate is transformed using sin(coord * PI).
     * @param {number[]} stateVector - The state vector to transform. Must be an array of numbers.
     * @returns {number[]} The transformed state vector. A new array is created.
     * @throws {Error} If the stateVector is invalid.
     */
    transformState(stateVector) {
        if (!Array.isArray(stateVector)) {
            throw new Error("State vector must be an array.");
        }

        const transformedVector = new Array(stateVector.length);
        for (let i = 0; i < stateVector.length; i++) {
            const coord = stateVector[i];
            if (typeof coord !== 'number') {
                throw new Error("State vector must contain only numbers.");
            }
            transformedVector[i] = Math.sin(coord * Math.PI);
        }
        return transformedVector;
    }

    /**
     * Applies a hyperbolic mapping (sinh function) to a state vector.
     * Each coordinate is transformed using sinh(coord).
     * @param {number[]} stateVector - The state vector to transform. Must be an array of numbers.
     * @returns {number[]} The transformed state vector.  A new array is created.
     * @throws {Error} If the stateVector is invalid.
     */
    applyHyperbolicMapping(stateVector) {
        if (!Array.isArray(stateVector)) {
            throw new Error("State vector must be an array.");
        }

        const transformedVector = new Array(stateVector.length);
        for (let i = 0; i < stateVector.length; i++) {
            const coord = stateVector[i];
            if (typeof coord !== 'number') {
                throw new Error("State vector must contain only numbers.");
            }
            transformedVector[i] = Math.sinh(coord);
        }
        return transformedVector;
    }

    /**
     * Applies a Riemannian transformation (contraction) to a state vector.
     * Each coordinate is transformed using coord / (1 + Math.abs(coord)).
     * @param {number[]} stateVector - The state vector to transform. Must be an array of numbers.
     * @returns {number[]} The transformed state vector. A new array is created.
     * @throws {Error} If the stateVector is invalid.
     */
    applyRiemannianTransformation(stateVector) {
        if (!Array.isArray(stateVector)) {
            throw new Error("State vector must be an array.");
        }

        const transformedVector = new Array(stateVector.length);
        for (let i = 0; i < stateVector.length; i++) {
            const coord = stateVector[i];
            if (typeof coord !== 'number') {
                throw new Error("State vector must contain only numbers.");
            }
            transformedVector[i] = coord / (1 + Math.abs(coord));
        }
        return transformedVector;
    }
}