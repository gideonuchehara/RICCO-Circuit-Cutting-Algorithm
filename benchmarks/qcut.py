import pennylane as qml
from pennylane import numpy as np
from typing import Tuple
from ricco.utils import generate_random_circuit, vqe_circuit, ricco_string_to_pauli_word

def qcut_random_unitary_expval(num_qubits: int, num_cuts: int, seed_u: int, seed_v: int, draw: bool = False) -> Tuple[float, float]:
    """
    Computes the expectation value of a quantum circuit using the PennyLane circuit-cutting 
    approach, which implements the approach outlined in Theorem 2 of Peng et al.
    Strategic placement of WireCut operations allows a quantum circuit to be split into 
    disconnected circuit fragments. Each circuit fragment is then executed multiple times 
    by varying the state preparations and measurements at incoming and outgoing cut locations, 
    respectively, resulting in a process tensor describing the action of the fragment. 
    The process tensors are then contracted to provide the result of the original uncut circuit.

    Args:
        num_qubits (int): Number of qubits in the quantum circuit.
        num_cuts (int): Number of qubits to cut in the circuit.
        seed_u (int): Seed for random generation of unitaries for subcircuits.
        seed_v (int): Seed for random generation of unitaries for subcircuits.
        draw (bool): If True, displays visual representations of the circuit before and after cutting.

    Returns:
        Tuple[float, float]: The RICCO expectation value of the reconstructed circuit and the 
        expectation value of the uncut circuit for comparison.
    """
    # Define RICCO parameters for unitary rotation at cut locations
    # params are not necessary in qcut.
    params = np.random.uniform(-np.pi, np.pi, size=4**num_cuts - 1, requires_grad=True)
    
    # Initialize device for RICCO optimization
    dev = qml.device("default.qubit", wires=range(num_qubits))
    
    # Define and create a QNode for the generated quantum circuit
    random_circuit_qnode = qml.QNode(generate_random_circuit, device=dev)
    
    # Update QNode and Compute expectation value of the uncut circuit
    uncut_random_circuit_expval = random_circuit_qnode(
        num_qubits, params, num_cuts, seed_u, seed_v, unitary_rotation=False
    )
    
    # Optionally display the uncut circuit
    if draw:
        fig1, ax = qml.draw_mpl(random_circuit_qnode)(num_qubits, params, num_cuts, 
                                                      seed_u, seed_v, unitary_rotation=False)
    
    # Convert the uncut QNode to a quantum tape for processing
    tape = random_circuit_qnode.qtape

    # Perform circuit cutting on the quantum tape
    graph = qml.transforms.qcut.tape_to_graph(tape)
    qml.transforms.qcut.replace_wire_cut_nodes(graph)

    # Generate fragments and the communication graph for circuit cutting
    fragments, communication_graph = qml.transforms.qcut.fragment_graph(graph)
    fragment_tapes = [qml.transforms.qcut.graph_to_tape(f) for f in fragments]
    
  
    # Define a device for running the new circuit fragments
    dev = qml.device("default.qubit", wires=len(fragment_tapes[0].wires))

    # Remap tape wires to align with the device wires
    fragment_tapes = [qml.map_wires(t, dict(zip(t.wires, dev.wires))) for t in fragment_tapes]
    
    # Expand each tape for post-processing via RICCO tensor network contraction
    expanded = [qml.transforms.qcut.expand_fragment_tape(t) for t in fragment_tapes]

    # Prepare configuration lists for contraction
    configurations = []
    prepare_nodes = []
    measure_nodes = []
    for tapes, p, m in expanded:
        configurations.append(tapes)
        prepare_nodes.append(p)
        measure_nodes.append(m)

    # Flatten expanded tapes for execution
    tapes = tuple(tape for c in configurations for tape in c)
    
    # Execute the tapes and process results with RICCO tensor contraction
    results = qml.execute(tapes, dev, gradient_fn=None)
    qcut_expectation = qml.transforms.qcut.qcut_processing_fn(
        results,
        communication_graph,
        prepare_nodes,
        measure_nodes,
    )
    
    return np.array(qcut_expectation), uncut_random_circuit_expval




def qcut_vqe_expval(num_qubits: int, num_cuts: int, vqe_params, hamiltonian, draw: bool = False) -> Tuple[float, float]:
    """
    Computes the expectation value of a quantum circuit using the RICCO circuit-cutting approach, 
    which optimizes unitary rotations at cut locations. This function generates, cuts, optimizes, 
    and reconstructs the quantum circuit.

    Args:
        num_qubits (int): Number of qubits in the quantum circuit.
        num_cuts (int): Number of qubits to cut in the circuit.
        seed_u (int): Seed for random generation of unitaries for subcircuits.
        seed_v (int): Seed for random generation of unitaries for subcircuits.
        draw (bool): If True, displays visual representations of the circuit before and after cutting.

    Returns:
        Tuple[float, float]: The RICCO expectation value of the reconstructed circuit and the 
        expectation value of the uncut circuit for comparison.
    """
    # Define RICCO parameters for unitary rotation at cut locations
    ricco_params = np.random.uniform(-np.pi, np.pi, size=4**num_cuts - 1, requires_grad=True)
    
    # Initialize device for RICCO optimization
    dev = qml.device("default.qubit", wires=range(num_qubits))
    
    # Define and create a QNode for the generated quantum circuit
    vqe_circuit_qnode = qml.QNode(vqe_circuit, device=dev)
    
    pauli_strings = [qml.grouping.pauli_word_to_string(P, wire_map=range(4)) for 
             P in hamiltonian.ops]
    observables = [ricco_string_to_pauli_word(word) for word in pauli_strings]
    
    H_coeff = hamiltonian.coeffs
    
    expectation_vals = []
    
    uncut_expectation_vals = []
    
    for obs in observables:
        print("obs = ", obs)
    
        # Update QNode and Compute expectation value of the uncut circuit
        uncut_vqe_circuit_expval = vqe_circuit_qnode(num_qubits, vqe_params, ricco_params, 
                                             obs, unitary_rotation=False)
        
        uncut_expectation_vals.append(float(uncut_vqe_circuit_expval))

        # Optionally display the uncut circuit
        if draw:
            fig1, ax = qml.draw_mpl(vqe_circuit_qnode)(num_qubits, vqe_params, ricco_params, 
                                             obs, unitary_rotation=False)

        # Convert the uncut QNode to a quantum tape for processing
        tape = vqe_circuit_qnode.qtape

        # Perform circuit cutting on the quantum tape
        graph = qml.transforms.qcut.tape_to_graph(tape)
        qml.transforms.qcut.replace_wire_cut_nodes(graph)
        

        new_fragments, new_communication_graph = qml.transforms.qcut.fragment_graph(graph)
        new_fragment_tapes = [qml.transforms.qcut.graph_to_tape(f) for f in new_fragments]

        # Define a device for running the new circuit fragments
        dev = qml.device("default.qubit", wires=range(max(len(new_fragment_tapes[0].wires),
                                                   len(new_fragment_tapes[1].wires))))

        # Remap tape wires to align with the device wires
        new_fragment_tapes = [qml.map_wires(t, dict(zip(t.wires, dev.wires))) for t in new_fragment_tapes]

        # Expand each tape for post-processing via RICCO tensor network contraction
        expanded = [qml.transforms.qcut.expand_fragment_tape(t) for t in new_fragment_tapes]

        # Prepare configuration lists for contraction
        configurations, prepare_nodes, measure_nodes = [], [], []
        for ext_tapes, p, m in expanded:
            configurations.append(ext_tapes)
            prepare_nodes.append(p)
            measure_nodes.append(m)

        # Flatten expanded tapes for execution
        expanded_tapes = tuple(t for c in configurations for t in c)        
        

        # Execute the tapes and process results with RICCO tensor contraction
        results = qml.execute(expanded_tapes, dev, gradient_fn=None)
        print("results = ", results)
        exp_val = qml.transforms.qcut.qcut_processing_fn(
            results, new_communication_graph, prepare_nodes, measure_nodes
        )
        
        expectation_vals.append(exp_val)
        print()
        
    print("pauli_strings = ", pauli_strings)
    qcut_expectation = np.sum(H_coeff*np.array(expectation_vals))
    print("expectation_vals = ", expectation_vals)
    uncut_circuit_expval = np.sum(H_coeff*np.array(uncut_expectation_vals))
    print("uncut_expectation_vals = ", uncut_expectation_vals)
    
    return np.array(qcut_expectation), uncut_circuit_expval
