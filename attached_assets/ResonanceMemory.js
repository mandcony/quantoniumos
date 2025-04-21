export class ResonanceMemory {
    constructor(frequency = 1, dampingFactor = 0.98) {
        this.frequency = frequency;
        this.phase = 0;
        this.dampingFactor = dampingFactor;
        this.memoryStore = {};
        this.feedbackLoop = {}; // Stores feedback-enhanced states
    }

    encode(data) {
        let phaseKey = this.phase % 360;
        this.memoryStore[phaseKey] = data;
        this.applyFeedback(phaseKey);
        this.phase = (this.phase + this.frequency) % 360;
        this.applyDamping();
    }

    applyFeedback(phaseKey) {
        // Resonance feedback loop to reinforce states
        if (!this.feedbackLoop[phaseKey]) {
            this.feedbackLoop[phaseKey] = this.memoryStore[phaseKey];
        } else {
            this.feedbackLoop[phaseKey] = (this.feedbackLoop[phaseKey] + this.memoryStore[phaseKey]) / 2;
        }
    }

    applyDamping() {
        Object.keys(this.memoryStore).forEach(phase => {
            this.memoryStore[phase] *= this.dampingFactor;
        });
    }

    recall(phaseOffset = 0) {
        let targetPhase = (this.phase + phaseOffset) % 360;
        return this.feedbackLoop[targetPhase] || this.memoryStore[targetPhase] || null;
    }
}
