export class QuantumBenchmark {
    constructor() {
        this.records = [];
    }

    logExecutionTime(label, startTime) {
        let duration = performance.now() - startTime;
        this.records.push({ label, duration });
        console.log(`[BENCHMARK] ${label}: ${duration.toFixed(4)}ms`);
    }

    analyzeQuantumFidelity(expected, actual) {
        let diff = expected.map((val, index) => Math.abs(val - actual[index]));
        let avgDiff = diff.reduce((a, b) => a + b, 0) / diff.length;
        console.log(`[FIDELITY] Average Deviation: ${avgDiff.toFixed(6)}`);
    }
}
