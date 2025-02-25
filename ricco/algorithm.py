import pennylane as qml
from pennylane import numpy as np
from typing import Tuple
from .utils import (generate_random_circuit, replace_measure_with_unitary, 
                   vqe_circuit, generate_n_qubit_unitary, ricco_string_to_pauli_word,
                  get_upstream_subcircuit)
from .optimization import ricco_unitary_optimization
from .fragmentation import ricco_expand_fragment_tape, ricco_processing_fn, ricco_fragment_graph

def ricco_random_unitary_expval(num_qubits: int, num_cuts: int, seed_u: int, seed_v: int, draw: bool = False) -> Tuple[float, float]:
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
        fig1, ax = qml.draw_mpl(random_circuit_qnode)(num_qubits, params, 
                                                      num_cuts, seed_u, seed_v, 
                                                      unitary_rotation=False)
    
    # Convert the uncut QNode to a quantum tape for processing
    tape = random_circuit_qnode.qtape

    # Perform circuit cutting on the quantum tape
    graph = qml.transforms.qcut.tape_to_graph(tape)
    qml.transforms.qcut.replace_wire_cut_nodes(graph)

    # Generate fragments and the communication graph for circuit cutting
    fragments, communication_graph = qml.transforms.qcut.fragment_graph(graph)
    fragment_tapes = [qml.transforms.qcut.graph_to_tape(f) for f in fragments]
    
    # Identify the upstream subcircuit with the unitary rotation to optimize
    upstream_subcircuit = replace_measure_with_unitary(fragment_tapes, params, generate_n_qubit_unitary)

    # Optionally display the upstream subcircuit
    if draw:
        print(qml.drawer.tape_mpl(upstream_subcircuit))
    
    # Optimize the RICCO unitary rotation at the cut location
    optimized_params, optimal_params, cost = ricco_unitary_optimization(fragment_tapes, params)
    
    # Update the QNode with optimized parameters and calculate the expectation value
    updated_random_circuit = random_circuit_qnode(
        num_qubits, optimized_params, num_cuts, seed_u, seed_v, unitary_rotation=True
    )

    # Convert the updated QNode to a quantum tape for further cutting
    updated_random_circuit_tape = random_circuit_qnode.qtape

    # Perform circuit cutting on the updated tape
    updated_graph = qml.transforms.qcut.tape_to_graph(updated_random_circuit_tape)
    qml.transforms.qcut.replace_wire_cut_nodes(updated_graph)

    new_fragments, new_communication_graph = qml.transforms.qcut.fragment_graph(updated_graph)
    new_fragment_tapes = [qml.transforms.qcut.graph_to_tape(f) for f in new_fragments]
    
    # Define a device for running the new circuit fragments
    dev = qml.device("default.qubit", wires=len(new_fragment_tapes[0].wires))

    # Remap tape wires to align with the device wires
    new_fragment_tapes = [qml.map_wires(t, dict(zip(t.wires, dev.wires))) for t in new_fragment_tapes]
    
    # Expand each tape for post-processing via RICCO tensor network contraction
    expanded = [ricco_expand_fragment_tape(t) for t in new_fragment_tapes]

    # Prepare configuration lists for contraction
    configurations, prepare_nodes, measure_nodes = [], [], []
    for ext_tapes, p, m, eig in expanded:
        configurations.append(ext_tapes)
        prepare_nodes.append(p)
        measure_nodes.append(m)

    # Flatten expanded tapes for execution
    expanded_tapes = tuple(t for c in configurations for t in c)
    
    # Execute the tapes and process results with RICCO tensor contraction
    results = qml.execute(expanded_tapes, dev, gradient_fn=None)
    ricco_expectation = ricco_processing_fn(
        results, new_communication_graph, prepare_nodes, measure_nodes
    )
    
    return np.array(ricco_expectation), uncut_random_circuit_expval





def uncut_expval(num_qubits: int, num_cuts: int, seed_u: int, seed_v: int, draw: bool = False) -> Tuple[float, float]:
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
    params = np.random.uniform(-np.pi, np.pi, size=4**num_cuts - 1, requires_grad=True)
    
    # Initialize device for RICCO optimization
    dev = qml.device("default.qubit", wires=range(num_qubits))
    
    # Define and create a QNode for the generated quantum circuit
    random_circuit_qnode = qml.QNode(generate_random_circuit, device=dev)
    
    # Compute expectation value of the uncut circuit
    uncut_random_circuit_expval = random_circuit_qnode(
        num_qubits, params, num_cuts, seed_u, seed_v, unitary_rotation=False
    )
    
    # Optionally display the uncut circuit
    if draw:
        fig1, ax = qml.draw_mpl(random_circuit_qnode)(num_qubits, params, 
                                                      num_cuts, seed_u, seed_v, 
                                                      unitary_rotation=False)
    
    
    return uncut_random_circuit_expval




# def ricco_vqe_expval(num_qubits: int, num_cuts: int, vqe_params, hamiltonian, draw: bool = False) -> Tuple[float, float]:
#     """
#     Computes the expectation value of a quantum circuit using the RICCO circuit-cutting approach, 
#     which optimizes unitary rotations at cut locations. This function generates, cuts, optimizes, 
#     and reconstructs the quantum circuit.

#     Args:
#         num_qubits (int): Number of qubits in the quantum circuit.
#         num_cuts (int): Number of qubits to cut in the circuit.
#         seed_u (int): Seed for random generation of unitaries for subcircuits.
#         seed_v (int): Seed for random generation of unitaries for subcircuits.
#         draw (bool): If True, displays visual representations of the circuit before and after cutting.

#     Returns:
#         Tuple[float, float]: The RICCO expectation value of the reconstructed circuit and the 
#         expectation value of the uncut circuit for comparison.
#     """
#     # Define RICCO parameters for unitary rotation at cut locations
#     ricco_params = np.random.uniform(-np.pi, np.pi, size=4**num_cuts - 1, requires_grad=True)
    
#     # Initialize device for RICCO optimization
#     dev = qml.device("default.qubit", wires=range(num_qubits))
    
#     # Define and create a QNode for the generated quantum circuit
#     vqe_circuit_qnode = qml.QNode(vqe_circuit, device=dev)
    
#     # Update QNode and Compute expectation value of the uncut circuit
#     uncut_vqe_circuit_expval = vqe_circuit_qnode(num_qubits, vqe_params, ricco_params, 
#                                          hamiltonian, unitary_rotation=False)
    
#     # Optionally display the uncut circuit
#     if draw:
#         fig1, ax = qml.draw_mpl(vqe_circuit_qnode)(num_qubits, vqe_params, ricco_params, 
#                                          hamiltonian, unitary_rotation=False)
    
#     # Convert the uncut QNode to a quantum tape for processing
#     tape = vqe_circuit_qnode.qtape

#     # Perform circuit cutting on the quantum tape
#     graph = qml.transforms.qcut.tape_to_graph(tape)
#     qml.transforms.qcut.replace_wire_cut_nodes(graph)

#     # Generate fragments and the communication graph for circuit cutting
#     # fragments, communication_graph = qml.transforms.qcut.fragment_graph(graph)
#     fragments, communication_graph = ricco_fragment_graph(graph)
#     fragment_tapes = [qml.transforms.qcut.graph_to_tape(f) for f in fragments]
    
#     # Note for Hamiltonians Pennylane does not have the functionality to fragment tapes
    
#     print("len(fragment_tapes) = ", len(fragment_tapes))
    
#     # Identify the upstream subcircuit with the unitary rotation to optimize
#     upstream_subcircuit = replace_measure_with_unitary(fragment_tapes, ricco_params, generate_n_qubit_unitary)

#     # Optionally display the upstream subcircuit
#     if draw:
#         print(qml.drawer.tape_mpl(upstream_subcircuit))
    
#     # Optimize the RICCO unitary rotation at the cut location
#     optimized_params, optimal_params, cost = ricco_unitary_optimization(fragment_tapes, ricco_params)
    
#     # Update the QNode with optimized parameters and calculate the expectation value
#     updated_vqe_circuit = vqe_circuit_qnode(num_qubits, vqe_params, ricco_params, 
#                                          hamiltonian, unitary_rotation=True)

#     # Convert the updated QNode to a quantum tape for further cutting
#     updated_vqe_circuit_tape = vqe_circuit_qnode.qtape

#     # Perform circuit cutting on the updated tape
#     updated_graph = qml.transforms.qcut.tape_to_graph(updated_vqe_circuit_tape)
#     qml.transforms.qcut.replace_wire_cut_nodes(updated_graph)

#     # new_fragments, new_communication_graph = qml.transforms.qcut.fragment_graph(updated_graph)
#     new_fragments, new_communication_graph = ricco_fragment_graph(updated_graph)
#     new_fragment_tapes = [qml.transforms.qcut.graph_to_tape(f) for f in new_fragments]
    
#     # Define a device for running the new circuit fragments
#     dev = qml.device("default.qubit", wires=range(max(len(new_fragment_tapes[0].wires),
#                                                len(new_fragment_tapes[1].wires))))

#     # Remap tape wires to align with the device wires
#     new_fragment_tapes = [qml.map_wires(t, dict(zip(t.wires, dev.wires))) for t in new_fragment_tapes]
    
#     # Expand each tape for post-processing via RICCO tensor network contraction
#     expanded = [ricco_expand_fragment_tape(t) for t in new_fragment_tapes]

#     # Prepare configuration lists for contraction
#     configurations, prepare_nodes, measure_nodes = [], [], []
#     for ext_tapes, p, m, eig in expanded:
#         configurations.append(ext_tapes)
#         prepare_nodes.append(p)
#         measure_nodes.append(m)

#     # Flatten expanded tapes for execution
#     expanded_tapes = tuple(t for c in configurations for t in c)
    
#     # Execute the tapes and process results with RICCO tensor contraction
#     results = qml.execute(expanded_tapes, dev, gradient_fn=None)
#     expectation_vals = ricco_processing_fn(
#         results, new_communication_graph, prepare_nodes, measure_nodes
#     )
    
#     H_coeff = hamiltonian.coeffs
#     ricco_expectation = np.sum(H_coeff*expectation_vals)
    
#     return np.array(ricco_expectation), uncut_random_circuit_expval



def ricco_vqe_expval(num_qubits: int, num_cuts: int, vqe_params, hamiltonian, draw: bool = False) -> Tuple[float, float]:
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
    # # Define RICCO parameters for unitary rotation at cut locations
    # ricco_params = np.random.uniform(-np.pi, np.pi, size=4**num_cuts - 1, requires_grad=True)
    
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
        # Define RICCO parameters for unitary rotation at cut locations
        ricco_params = np.random.uniform(-np.pi, np.pi, size=4**num_cuts - 1, requires_grad=True)
    
        # Update QNode and Compute expectation value of the uncut circuit
        uncut_vqe_circuit_expval = vqe_circuit_qnode(num_qubits, vqe_params, ricco_params, 
                                             obs, unitary_rotation=False)
        
        # uncut_expectation_vals.append(float(uncut_vqe_circuit_expval))

        # Optionally display the uncut circuit
        if draw:
            fig1, ax = qml.draw_mpl(vqe_circuit_qnode)(num_qubits, vqe_params, ricco_params, 
                                             obs, unitary_rotation=False)

        # Convert the uncut QNode to a quantum tape for processing
        tape = vqe_circuit_qnode.qtape

        # Perform circuit cutting on the quantum tape
        graph = qml.transforms.qcut.tape_to_graph(tape)
        qml.transforms.qcut.replace_wire_cut_nodes(graph)

        # Generate fragments and the communication graph for circuit cutting
        # fragments, communication_graph = qml.transforms.qcut.fragment_graph(graph)
        # print("len(fragments) = ", len(fragments))
        # this was modified to account for measurement observables that are identity
        fragments, communication_graph = ricco_fragment_graph(graph)
        fragment_tapes = [qml.transforms.qcut.graph_to_tape(f) for f in fragments]

        # Note for Hamiltonians Pennylane does not have the functionality to fragment tapes

        # print("len(fragment_tapes) = ", len(fragment_tapes))

        # Identify the upstream subcircuit with the unitary rotation to optimize
        upstream_subcircuit = replace_measure_with_unitary(fragment_tapes, ricco_params, generate_n_qubit_unitary)

        # Optionally display the upstream subcircuit
        if draw:
            print(qml.drawer.tape_mpl(upstream_subcircuit))

        # Optimize the RICCO unitary rotation at the cut location
        optimized_params, optimal_params, cost = ricco_unitary_optimization(fragment_tapes, ricco_params)

        # Update the QNode with optimized parameters and calculate the expectation value
        updated_vqe_circuit = vqe_circuit_qnode(num_qubits, vqe_params, optimized_params, 
                                             obs, unitary_rotation=True)
        
        uncut_expectation_vals.append(float(updated_vqe_circuit))
        
        # if draw:
        #     fig1, ax = qml.draw_mpl(vqe_circuit_qnode)(num_qubits, vqe_params, ricco_params, 
        #                                      obs, unitary_rotation=True)

        # Convert the updated QNode to a quantum tape for further cutting
        updated_vqe_circuit_tape = vqe_circuit_qnode.qtape

        # Perform circuit cutting on the updated tape
        updated_graph = qml.transforms.qcut.tape_to_graph(updated_vqe_circuit_tape)
        qml.transforms.qcut.replace_wire_cut_nodes(updated_graph)

        new_fragments, new_communication_graph = qml.transforms.qcut.fragment_graph(updated_graph)
        # new_fragments, new_communication_graph = ricco_fragment_graph(updated_graph)
        new_fragment_tapes = [qml.transforms.qcut.graph_to_tape(f) for f in new_fragments]

        # Define a device for running the new circuit fragments
        dev = qml.device("default.qubit", wires=range(max(len(new_fragment_tapes[0].wires),
                                                   len(new_fragment_tapes[1].wires))))

        # Remap tape wires to align with the device wires
        new_fragment_tapes = [qml.map_wires(t, dict(zip(t.wires, dev.wires))) for t in new_fragment_tapes]

        # Expand each tape for post-processing via RICCO tensor network contraction
        expanded = [ricco_expand_fragment_tape(t) for t in new_fragment_tapes]

        # Prepare configuration lists for contraction
        configurations, prepare_nodes, measure_nodes = [], [], []
        for ext_tapes, p, m, eig in expanded:
            configurations.append(ext_tapes)
            prepare_nodes.append(p)
            measure_nodes.append(m)

        # Flatten expanded tapes for execution
        expanded_tapes = tuple(t for c in configurations for t in c)

        # Execute the tapes and process results with RICCO tensor contraction
        results = qml.execute(expanded_tapes, dev, gradient_fn=qml.gradients.param_shift)
        # pennylane does not implement the measurement of the identity. Here we
        # identify where the identity observable was measure and replace it with
        # 1 to indicate the expectation value of the identity operator which is 1
        # results = [qml.math.array([1]) if arr.size == 0 else arr for arr in results]
        # results = [arr for arr in results if arr.size > 0]
        print("results = ", results)
        exp_val = ricco_processing_fn(
            results, new_communication_graph, prepare_nodes, measure_nodes
        )
        
        expectation_vals.append(exp_val)
        print()
        
    print("pauli_strings = ", pauli_strings)
    ricco_expectation = np.sum(H_coeff*np.array(expectation_vals))
    print("expectation_vals = ", expectation_vals)
    uncut_circuit_expval = np.sum(H_coeff*np.array(uncut_expectation_vals))
    print("uncut_expectation_vals = ", uncut_expectation_vals)
    
    return np.array(ricco_expectation), uncut_circuit_expval






# def ricco_vqe_expval(num_qubits: int, num_cuts: int, vqe_params, hamiltonian, draw: bool = False) -> Tuple[float, float]:
#     """
#     Computes the expectation value of a quantum circuit using the RICCO circuit-cutting approach, 
#     which optimizes unitary rotations at cut locations. This function generates, cuts, optimizes, 
#     and reconstructs the quantum circuit.

#     Args:
#         num_qubits (int): Number of qubits in the quantum circuit.
#         num_cuts (int): Number of qubits to cut in the circuit.
#         seed_u (int): Seed for random generation of unitaries for subcircuits.
#         seed_v (int): Seed for random generation of unitaries for subcircuits.
#         draw (bool): If True, displays visual representations of the circuit before and after cutting.

#     Returns:
#         Tuple[float, float]: The RICCO expectation value of the reconstructed circuit and the 
#         expectation value of the uncut circuit for comparison.
#     """
#     # # Define RICCO parameters for unitary rotation at cut locations
#     # ricco_params = np.random.uniform(-np.pi, np.pi, size=4**num_cuts - 1, requires_grad=True)
    
#     # Initialize device for RICCO optimization
#     dev = qml.device("default.qubit", wires=range(num_qubits))
    
#     # Define and create a QNode for the generated quantum circuit
#     vqe_circuit_qnode = qml.QNode(vqe_circuit, device=dev)
    
#     # Define RICCO parameters for unitary rotation at cut locations
#     ricco_params = np.random.uniform(-np.pi, np.pi, size=4**num_cuts - 1, requires_grad=True)
    
#     # Get the upstream circuit from the vqe_circuit
#     upstream_subcircuit, downstream_subcircuit = get_upstream_subcircuit(vqe_circuit, num_qubits, num_cuts, vqe_params, ricco_params, 
#                                                  hamiltonian, unitary_rotation=False)
    
#     # print(any(isinstance(op, qml.transforms.qcut.MeasureNode) for op in upstream_subcircuit.operations))
    
#     # Identify and replace measureNode in the upstream subcircuit with the unitary rotation to optimize
#     upstream_subcircuit = replace_measure_with_unitary([upstream_subcircuit, downstream_subcircuit], ricco_params, generate_n_qubit_unitary)

#     # Optimize the RICCO unitary rotation at the cut location
#     optimized_params, optimal_params, cost = ricco_unitary_optimization(upstream_subcircuit, ricco_params)
    
    
    
#     pauli_strings = [qml.grouping.pauli_word_to_string(P, wire_map=range(4)) for 
#              P in hamiltonian.ops]
#     observables = [ricco_string_to_pauli_word(word) for word in pauli_strings]
    
#     H_coeff = hamiltonian.coeffs
    
#     expectation_vals = []
    
#     uncut_expectation_vals = []
    
    
    
#     for obs in observables:
#         print("obs = ", obs)
        
#         # Update QNode and Compute expectation value of the uncut circuit
#         uncut_vqe_circuit_expval = vqe_circuit_qnode(num_qubits, vqe_params, ricco_params, 
#                                              obs, unitary_rotation=False)
        
       
#         # Update the QNode with optimized parameters and calculate the expectation value
#         updated_vqe_circuit = vqe_circuit_qnode(num_qubits, vqe_params, optimized_params, 
#                                              obs, unitary_rotation=True)
        
#         uncut_expectation_vals.append(float(updated_vqe_circuit))
        
#         # if draw:
#         #     fig1, ax = qml.draw_mpl(vqe_circuit_qnode)(num_qubits, vqe_params, ricco_params, 
#         #                                      obs, unitary_rotation=True)

#         # Convert the updated QNode to a quantum tape for further cutting
#         updated_vqe_circuit_tape = vqe_circuit_qnode.qtape

#         # Perform circuit cutting on the updated tape
#         updated_graph = qml.transforms.qcut.tape_to_graph(updated_vqe_circuit_tape)
#         qml.transforms.qcut.replace_wire_cut_nodes(updated_graph)

#         # new_fragments, new_communication_graph = qml.transforms.qcut.fragment_graph(updated_graph)
#         new_fragments, new_communication_graph = ricco_fragment_graph(updated_graph)
#         new_fragment_tapes = [qml.transforms.qcut.graph_to_tape(f) for f in new_fragments]

#         # Define a device for running the new circuit fragments
#         dev = qml.device("default.qubit", wires=range(max(len(new_fragment_tapes[0].wires),
#                                                    len(new_fragment_tapes[1].wires))))

#         # Remap tape wires to align with the device wires
#         new_fragment_tapes = [qml.map_wires(t, dict(zip(t.wires, dev.wires))) for t in new_fragment_tapes]

#         # Expand each tape for post-processing via RICCO tensor network contraction
#         expanded = [ricco_expand_fragment_tape(t) for t in new_fragment_tapes]

#         # Prepare configuration lists for contraction
#         configurations, prepare_nodes, measure_nodes = [], [], []
#         for ext_tapes, p, m, eig in expanded:
#             configurations.append(ext_tapes)
#             prepare_nodes.append(p)
#             measure_nodes.append(m)

#         # Flatten expanded tapes for execution
#         expanded_tapes = tuple(t for c in configurations for t in c)

#         # Execute the tapes and process results with RICCO tensor contraction
#         results = qml.execute(expanded_tapes, dev, gradient_fn=None)
#         # pennylane does not implement the measurement of the identity. Here we
#         # identify where the identity observable was measure and replace it with
#         # 1 to indicate the expectation value of the identity operator which is 1
#         # results = [qml.math.array([1]) if arr.size == 0 else arr for arr in results]
#         # results = [arr for arr in results if arr.size > 0]
#         print("results = ", results)
#         exp_val = ricco_processing_fn(
#             results, new_communication_graph, prepare_nodes, measure_nodes
#         )
        
#         expectation_vals.append(exp_val)
#         print()
        
#     print("pauli_strings = ", pauli_strings)
#     ricco_expectation = np.sum(H_coeff*np.array(expectation_vals))
#     print("expectation_vals = ", expectation_vals)
#     uncut_circuit_expval = np.sum(H_coeff*np.array(uncut_expectation_vals))
#     print("uncut_expectation_vals = ", uncut_expectation_vals)
    
#     return np.array(ricco_expectation), uncut_circuit_expval