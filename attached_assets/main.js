// main.js
import GeometricContainer from './services/GeometricContainer';
import LinearRegion from './services/LinearRegion';
import Shard from './services/Shard';
import QuantumSearch from './services/QuantumSearch';

// Example material properties
const containerMaterial = {
  youngsModulus: 1e9,
  density: 2700
};

// Container 1
const container1Vertices = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]];
const container1 = new GeometricContainer('container1', container1Vertices, [
  { rotation: { x: Math.PI / 4 } },
  { scale: { x: 1.2, y: 0.8 } },
  { translation: { x: 0.5, y: -0.5, z: 0.2 } }
], containerMaterial);
container1.encodeData("101");

// Container 2
const container2Vertices = [[2, 2, 0], [3, 2, 0], [3, 3, 0], [2, 3, 0]];
const container2 = new GeometricContainer('container2', container2Vertices, [
  { rotation: { y: Math.PI / 2 } },
  { scale: { x: 0.7, y: 1.3 } },
  { translation: { x: -0.3, y: 0.7, z: 0.1 } }
], containerMaterial);
container2.encodeData("010");

// Linear regions
const lr1 = new LinearRegion(container1Vertices);
const lr2 = new LinearRegion(container2Vertices);
container1.addLinearRegion(lr1);
container2.addLinearRegion(lr2);

container1.calculateResonantFrequencies(0.1);
container2.calculateResonantFrequencies(0.1);

// Build a shard
const shard = new Shard([container1, container2]);

// Example resonance check
const targetFreq = container1.resonantFrequencies[0];
const results = shard.search(targetFreq, 0.1);
if (results.length > 0) {
  console.log("Found resonance in:", results.map(c => c.id));
} else {
  console.log("No resonance found.");
}

// If you want to do the quantum search approach:
const quantumSearch = new QuantumSearch();
const foundContainer = quantumSearch.search([container1, container2], targetFreq);
if (foundContainer) {
  console.log("QuantumSearch found container:", foundContainer.id);
} else {
  console.log("QuantumSearch: no container found.");
}
