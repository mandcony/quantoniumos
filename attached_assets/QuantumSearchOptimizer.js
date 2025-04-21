/**
 * QuantumSearchOptimizer: Simulates a quantum search algorithm (Grover's algorithm).
 *  This class provides methods to amplify the probability of finding a target element within a database.
 */
export class QuantumSearchOptimizer {
    /**
     * Creates a new QuantumSearchOptimizer.
     * @param {Array} database - The database to search within.  Must be a non-empty array.
     * @throws {Error} If the database is not a non-empty array.
     */
    constructor(database) {
        if (!Array.isArray(database) || database.length === 0) {
            throw new Error("Database must be a non-empty array.");
        }

        this.database = [...database]; // Create a defensive copy to prevent external modification
        this.size = database.length;
        this.phaseShift = Math.PI / this.size;  // Calculate the phase shift based on the database size (Grover's algorithm)
    }

    /**
     * Amplifies the probability of finding the element at the specified target index.
     * This is a core step in Grover's algorithm simulation.
     * @param {number} targetIndex - The index of the target element in the database. Must be a valid index within the database.
     * @returns {number[]} An array representing the probability amplitudes after the amplification step.
     * @throws {Error} If the targetIndex is out of bounds.
     * @private
     */
    amplifyProbability(targetIndex) {
        if (!Number.isInteger(targetIndex) || targetIndex < 0 || targetIndex >= this.size) {
            throw new Error("Target index is out of bounds.");
        }

        const inversion = new Array(this.size);
        for (let i = 0; i < this.size; i++) {
            inversion[i] = i === targetIndex ? Math.cos(this.phaseShift) : Math.sin(this.phaseShift);
        }
        return inversion;
    }

    /**
     * Searches for the target element in the database and, if found, amplifies its probability.
     * @param {*} target - The element to search for.
     * @returns {number[]|null} An array representing the probability amplitudes after amplification, or null if the target is not found.
     */
    search(target) {
        const index = this.database.indexOf(target);
        return index !== -1 ? this.amplifyProbability(index) : null;
    }

    /**
     * Performs an adaptive search, incorporating a random factor into the amplified result.
     * This simulates noise or uncertainty in the quantum search process.  It might deviate from true Grover's algorithm.
     * @param {*} target - The element to search for.
     * @returns {number[]|null} An array representing the adaptively amplified probability amplitudes, or null if the target is not found.
     */
    adaptiveSearch(target) {
        const amplifiedResult = this.search(target);
        if (!amplifiedResult) {
            return null;
        }

        const noisyResult = new Array(this.size);
        for (let i = 0; i < this.size; i++) {
            noisyResult[i] = amplifiedResult[i] * Math.random();
        }
        return noisyResult;
    }
}