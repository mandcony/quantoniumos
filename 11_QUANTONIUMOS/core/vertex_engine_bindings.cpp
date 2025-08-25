#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include "vertex_engine.cpp"

namespace py = pybind11;

PYBIND11_MODULE(vertex_engine, m) {
    m.doc() = "C++ Vertex Engine for Quantum Topology Operations";
    
    // QuantumVertex class
    py::class_<QuantumVertex>(m, "QuantumVertex")
        .def(py::init<int, double>(), py::arg("id"), py::arg("freq") = 1.0)
        .def_readwrite("alpha", &QuantumVertex::alpha)
        .def_readwrite("beta", &QuantumVertex::beta)
        .def_readwrite("frequency", &QuantumVertex::frequency)
        .def_readwrite("phase", &QuantumVertex::phase)
        .def_readwrite("vertex_id", &QuantumVertex::vertex_id)
        .def_readwrite("position", &QuantumVertex::position)
        .def("apply_pauli_x", &QuantumVertex::apply_pauli_x)
        .def("apply_pauli_y", &QuantumVertex::apply_pauli_y)
        .def("apply_pauli_z", &QuantumVertex::apply_pauli_z)
        .def("apply_hadamard", &QuantumVertex::apply_hadamard)
        .def("apply_phase_gate", &QuantumVertex::apply_phase_gate)
        .def("apply_rft_resonance", &QuantumVertex::apply_rft_resonance)
        .def("evolve_quantum_step", &QuantumVertex::evolve_quantum_step)
        .def("get_state_vector", &QuantumVertex::get_state_vector)
        .def("measure_probability_0", &QuantumVertex::measure_probability_0)
        .def("measure_probability_1", &QuantumVertex::measure_probability_1);
    
    // HarmonicOscillator class
    py::class_<HarmonicOscillator>(m, "HarmonicOscillator")
        .def(py::init<double, std::complex<double>>(), 
             py::arg("freq"), py::arg("amp") = std::complex<double>(1.0, 0.0))
        .def_readwrite("frequency", &HarmonicOscillator::frequency)
        .def_readwrite("amplitude", &HarmonicOscillator::amplitude)
        .def_readwrite("damping", &HarmonicOscillator::damping)
        .def("time_step", &HarmonicOscillator::time_step)
        .def("vibrational_mode", &HarmonicOscillator::vibrational_mode)
        .def("quantum_step", &HarmonicOscillator::quantum_step);
    
    // VertexNetwork class
    py::class_<VertexNetwork>(m, "VertexNetwork")
        .def(py::init<int>())
        .def_readwrite("vertices", &VertexNetwork::vertices)
        .def_readwrite("oscillators", &VertexNetwork::oscillators)
        .def_readwrite("num_vertices", &VertexNetwork::num_vertices)
        .def("initialize_vertices", &VertexNetwork::initialize_vertices)
        .def("create_oscillators", &VertexNetwork::create_oscillators)
        .def("create_network_connections", &VertexNetwork::create_network_connections)
        .def("apply_single_qubit_gate", &VertexNetwork::apply_single_qubit_gate,
             py::arg("vertex_id"), py::arg("gate"), py::arg("parameter") = 0.0)
        .def("apply_cnot_gate", &VertexNetwork::apply_cnot_gate)
        .def("evolve_quantum_network", &VertexNetwork::evolve_quantum_network,
             py::arg("time_steps") = 10, py::arg("dt") = 0.05)
        .def("get_all_state_vectors", &VertexNetwork::get_all_state_vectors)
        .def("get_probability_distribution", &VertexNetwork::get_probability_distribution)
        .def("get_total_entanglement", &VertexNetwork::get_total_entanglement)
        .def("get_num_vertices", &VertexNetwork::get_num_vertices)
        .def("get_num_edges", &VertexNetwork::get_num_edges);
    
    // Utility functions
    m.def("test_vertex_engine", &test_vertex_engine, "Test the C++ vertex engine");
    
    // Constants
    m.attr("PHI") = (1.0 + std::sqrt(5.0)) / 2.0;  // Golden ratio
}
