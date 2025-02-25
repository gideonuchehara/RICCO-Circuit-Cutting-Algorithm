import pennylane as qml
from pennylane import numpy as np
from pennylane.tape import QuantumTape
from itertools import product
from typing import List, Tuple
import random
from scipy.stats import unitary_group
import matplotlib.pyplot as plt
from pennylane.ops import Identity, PauliX, PauliY, PauliZ
from .fragmentation import ricco_fragment_graph


def ricco_string_to_pauli_word(pauli_string, wire_map=None):
    """NOTE: THIS FUNCTION IS NECESSARY TO ACCOUNT FOR IDENTITY OBSERVABLES IN CUTTING
    Convert a string in terms of ``'I'``, ``'X'``, ``'Y'``, and ``'Z'`` into a Pauli word
    for the given wire map.

    Args:
        pauli_string (str): A string of characters consisting of ``'I'``, ``'X'``, ``'Y'``, and ``'Z'``
            indicating a Pauli word.
        wire_map (dict[Union[str, int], int]): dictionary containing all wire labels used in
            the Pauli word as keys, and unique integer labels as their values

    Returns:
        .Observable: The Pauli word representing of ``pauli_string`` on the wires
        enumerated in the wire map.

    **Example**

    >>> wire_map = {'a' : 0, 'b' : 1, 'c' : 2}
    >>> string_to_pauli_word('XIY', wire_map=wire_map)
    PauliX(wires=['a']) @ PauliY(wires=['c'])
    """
    character_map = {"I": Identity, "X": PauliX, "Y": PauliY, "Z": PauliZ}

    if not isinstance(pauli_string, str):
        raise TypeError(f"Input to string_to_pauli_word must be string, obtained {pauli_string}")

    # String can only consist of I, X, Y, Z
    if any(char not in character_map for char in pauli_string):
        raise ValueError(
            "Invalid characters encountered in string_to_pauli_word "
            f"string {pauli_string}. Permitted characters are 'I', 'X', 'Y', and 'Z'"
        )

    # If no wire map is provided, construct one using integers based on the length of the string
    if wire_map is None:
        wire_map = {x: x for x in range(len(pauli_string))}

    if len(pauli_string) != len(wire_map):
        raise ValueError(
            "Wire map and pauli_string must have the same length to convert "
            "from string to Pauli word."
        )

    # # Special case: all-identity Pauli
    # if pauli_string == "I" * len(wire_map):
    #     first_wire = list(wire_map)[0]
    #     return Identity(first_wire)

    pauli_word = None

    for wire_name, wire_idx in wire_map.items():
        pauli_char = pauli_string[wire_idx]

        # # Don't care about the identity
        # if pauli_char == "I":
        #     continue

        if pauli_word is not None:
            pauli_word = pauli_word @ character_map[pauli_char](wire_name)
        else:
            pauli_word = character_map[pauli_char](wire_name)

    return pauli_word




def generate_random_circuit(num_qubits, params, num_cuts=1, 
                            seed_u=0, seed_v=0, unitary_rotation=False):
    """
    Generates a random unitary quantum circuit with specified qubits and cuts.
    
    Parameters:
    - num_qubits (int): Total number of qubits in the circuit.
    - num_cuts (int): Number of cut qubits between subcircuits.
    - seed_u (int): Seed for generating the random unitary in subcircuit AB.
    - seed_v (int): Seed for generating the random unitary in subcircuit BC.
    - unitary_rotation (bol): Insert unitary rotation at cut location if True
    
    Returns:
    - qml.expval: Expectation value of the observable on the entire circuit.
    
    Raises:
    - ValueError: If the number of cuts exceeds the allowable range.
    """
    
    # Divide qubits into three sections: A, B (cut qubits), and C
    N_A_wires = num_qubits // 2
    N_B_wires = num_cuts
    N_C_wires = num_qubits - (N_A_wires + N_B_wires)
    
    # Validate the number of cuts
    if num_cuts > N_B_wires + N_C_wires or num_cuts > num_qubits:
        raise ValueError("The number of cuts exceeds the allowable range based on the circuit's qubit count.")
    
    # Define wire labels for each section
    A_wires = [f"A{x}" for x in range(N_A_wires)]
    B_wires = [f"B{x}" for x in range(N_B_wires)]
    C_wires = [f"C{x}" for x in range(N_C_wires)]
    
    # Define wire mappings for each subcircuit
    AB_wires = A_wires + B_wires  # Wires for subcircuit AB
    BC_wires = B_wires + C_wires  # Wires for subcircuit BC
    
    # Main observable (Pauli Z on each qubit)
    observable = qml.grouping.string_to_pauli_word("Z" * num_qubits)
    
    # Generate random unitaries for subcircuits AB and BC
    U_ab = unitary_group.rvs(2 ** len(AB_wires), random_state=seed_u)
    U_bc = unitary_group.rvs(2 ** len(BC_wires), random_state=seed_v)
    
    # Apply unitaries and wire cut
    qml.QubitUnitary(U_ab, wires=range(len(AB_wires)))
    
    # Insert unitary rotation at cut location
    if unitary_rotation:
        generate_n_qubit_unitary(params, list(range(N_A_wires, N_A_wires + N_B_wires)))
        qml.WireCut(wires=range(N_A_wires, N_A_wires + N_B_wires))
        qml.adjoint(generate_n_qubit_unitary)(params, list(range(N_A_wires, N_A_wires + 
                                                                 N_B_wires)))
    else: # No unitary rotations inserted at cut location
        qml.WireCut(wires=range(N_A_wires, N_A_wires + N_B_wires))
    
    qml.QubitUnitary(U_bc, wires=range(N_A_wires, num_qubits))
    
    # Return expectation value of the observable
    return qml.expval(observable)




def generate_n_qubit_unitary(params, num_qubits):
    """
    Generates a general n-qubit unitary circuit using interleaved single-qubit rotations and controlled-ROT gates.
    
    Parameters:
    - params (list or np.ndarray): A flat list or array of parameters for the circuit gates.
    - num_qubits (int or list): The number of qubits (int) or a list of wire labels.
    
    Returns:
    - None: Applies gates directly within the PennyLane quantum node context.
    
    Raises:
    - ValueError: If the length of `params` does not match the theoretical number of parameters for an n-qubit unitary.
    - TypeError: If `num_qubits` is neither an integer nor a list of wire labels.
    
    Notes:
    - **Structure**: The ansatz consists of layers where:
      - Single-qubit rotations are applied to each qubit.
      - Controlled-ROT gates are applied between pairs of qubits in a linear chain if `num_qubits > 1`.
      - This structure can approximate arbitrary unitaries with enough layers.
      
    - **Parameter Count for Arbitrary n-Qubit Unitary**: 
      An arbitrary n-qubit unitary matrix \( U \) requires \( 2^{2n} - 1 \) real parameters.
    """
    
    # for testing
    if num_qubits==1:
        qml.Rot(*params, wires=num_qubits)
    
    # Check if `num_qubits` is an integer or list of wire labels
    if isinstance(num_qubits, int):
        wires = list(range(num_qubits))  # Default wire labels if `num_qubits` is an integer
    elif isinstance(num_qubits, list):
        wires = num_qubits  # Use provided list of wire labels
        num_qubits = len(wires)  # Update `num_qubits` to the length of the wire list
    else:
        raise TypeError("num_qubits must be an integer or a list of wire labels.")
    
    # Theoretical parameter count for an arbitrary n-qubit unitary
    expected_num_params = 2 ** (2 * num_qubits) - 1
    if len(params) != expected_num_params:
        raise ValueError(f"Expected {expected_num_params} parameters for a {num_qubits}-qubit unitary, but got {len(params)}.")
    
    param_offset = 0
    
    # Build the layered ansatz
    for layer in range(num_qubits):  # Adjusted number of layers
        # Apply single-qubit rotations for each qubit
        for i in range(num_qubits):
            # Check if param_offset is within bounds before applying rotations
            if param_offset + 2 >= len(params):
                return  # Stop if we've used all parameters
            qml.Rot(params[param_offset], params[param_offset + 1], params[param_offset + 2], wires=wires[i])
            param_offset += 3
        
        # Apply controlled-ROT gates in a linear chain
        for i in range(num_qubits - 1):
            # Check if param_offset is within bounds before applying controlled ROTs
            if param_offset + 2 >= len(params):
                return  # Stop if we've used all parameters
            qml.CRot(params[param_offset], params[param_offset + 1], params[param_offset + 2], wires=[wires[i], wires[i + 1]])
            param_offset += 3
        
        # Optional layer of additional single-qubit rotations
        for i in range(num_qubits):
            if param_offset + 2 >= len(params):
                return  # Stop if we've used all parameters
            qml.Rot(params[param_offset], params[param_offset + 1], params[param_offset + 2], wires=wires[i])
            param_offset += 3





def generate_observables(num_uncut_qubits: int, num_cut_qubits: int) -> List[str]:
    """
    Generates observables based on the specified pattern:
    - Uncut qubits are represented with "Z".
    - Cut qubits are represented with either "I" or "Z".
    
    Args:
        num_uncut_qubits (int): Number of uncut qubits, which will always have "Z" observables.
        num_cut_qubits (int): Number of cut qubits, which can have either "I" or "Z" observables.

    Returns:
        List[str]: A list of strings representing the generated observables.
    """
    uncut_observables = "Z" * num_uncut_qubits
    cut_combinations = product("IZ", repeat=num_cut_qubits)
    return [uncut_observables + ''.join(cut_combination) for cut_combination in cut_combinations]

def replace_measure_with_unitary(
    subcircuits: List[qml.tape.QuantumScript],
    params,
    generate_unitary
) -> QuantumTape:
    """
    Replaces MeasureNodes on cut qubits in the subcircuit containing MeasureNodes with
    a unitary rotation, and updates the measurements to be expectation values of the new observables.

    Args:
        subcircuits (List[QuantumScript]): List of subcircuits resulting from the circuit cut.
        params: Parameters for the unitary rotation on the cut qubits.
        generate_unitary (callable): Function to generate a unitary rotation, given params and qubit labels.

    Returns:
        QuantumTape: A new quantum tape with the unitary rotation applied to the cut qubits
                     in place of the MeasureNodes, and with updated measurement observables.

    Raises:
        ValueError: If neither subcircuit contains any MeasureNodes.
    """
    # Identify the subcircuit that contains MeasureNodes
    subcircuit_with_measure = None
    for subcircuit in subcircuits:
        # print(subcircuit.operations)
        # print(any(isinstance(op, qml.transforms.qcut.MeasureNode) for op in subcircuit.operations))
        # print()
        if any(isinstance(op, qml.transforms.qcut.MeasureNode) for op in subcircuit.operations):
            subcircuit_with_measure = subcircuit.copy()
            # print(subcircuit.operations)
            # print("I am here")
            break

    # print(subcircuit_with_measure.operations)
    # Raise an error if no subcircuit with MeasureNodes is found
    if subcircuit_with_measure is None:
        raise ValueError("No subcircuit contains MeasureNodes.")

    # Identify the other subcircuit (assumes exactly two subcircuits)
    subcircuit_other = subcircuits[1] if subcircuits[0] is subcircuit_with_measure else subcircuits[0]

    # Identify the cut qubits (common qubits between the two subcircuits)
    # cut_qubits = list(set(subcircuit_with_measure.wires) & set(subcircuit_other.wires))
    
    
    cut_qubits = [op.wires for op in subcircuit.operations if isinstance(op, qml.transforms.qcut.MeasureNode)]

    # Determine number of cut and uncut qubits
    num_uncut_qubits = len(subcircuit_with_measure.wires) - len(cut_qubits)
    num_cut_qubits = len(cut_qubits)
    print("num_cut_qubits = ", num_cut_qubits)
    
    

    # Generate new observables for measurement
    new_observables = generate_observables(num_uncut_qubits, num_cut_qubits)

    # Create a new QuantumTape to hold the modified operations
    with QuantumTape() as new_tape:
        for op in subcircuit_with_measure.operations:
            if isinstance(op, qml.transforms.qcut.MeasureNode) and op.wires[0] in cut_qubits:
                # Replace MeasureNode with the unitary rotation on the cut qubit
                qml.apply(generate_unitary(params, cut_qubits))
            elif not isinstance(op, qml.transforms.qcut.MeasureNode):
                # Apply the operation as it is if it's not a MeasureNode on a cut qubit
                qml.apply(op)

        # Add expectation values for each new observable to ensure measurements are defined
        for obs_str in new_observables:
            observable = qml.grouping.string_to_pauli_word(obs_str, wire_map={w: i for i, w in enumerate(subcircuit_with_measure.wires)})
            qml.expval(observable)

    return new_tape



def vqe_circuit(num_qubits, vqe_params, ricco_params, observable, unitary_rotation=False):
    qubits=list(range(num_qubits))
    
    qml.PauliX(wires=0)
    qml.PauliX(wires=1)
    for i in qubits:
        qml.RZ(vqe_params[3 * i], wires=i)
        qml.RY(vqe_params[3 * i + 1], wires=i)
        qml.RZ(vqe_params[3 * i + 2], wires=i)
    qml.CNOT(wires=[2, 3])
    
    # Insert unitary rotation at cut location
    if unitary_rotation:
        generate_n_qubit_unitary(ricco_params, [qubits[2]])
        qml.WireCut(wires=qubits[2])  # apply cut here
        qml.adjoint(generate_n_qubit_unitary)(ricco_params, [qubits[2]])
    else: # No unitary rotations inserted at cut location
        qml.WireCut(wires=qubits[2])  # apply cut here
        
    qml.CNOT(wires=[2, 0])
    qml.CNOT(wires=[3, 1])
    
    return qml.expval(observable)
    
#     pauli_strings = [qml.grouping.pauli_word_to_string(P, wire_map=range(4)) for 
#                  P in hamiltonian.ops]
#     observables = [qml.grouping.string_to_pauli_word(word) for word in pauli_strings]
    
#     return [qml.expval(obs) for obs in observables]
    

    
# from utils import vqe_circuit, ricco_string_to_pauli_word
# from pennylane import tape

def get_upstream_subcircuit(vqe_circuit, num_qubits, num_cuts, vqe_params, ricco_params, 
                                                 hamiltonian, unitary_rotation=False):
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
    observable_list = []

    for obs in observables:
        # print("obs = ", obs)
        # Define RICCO parameters for unitary rotation at cut locations
        ricco_params = np.random.uniform(-np.pi, np.pi, size=4**num_cuts - 1, requires_grad=True)

        # Update QNode and Compute expectation value of the uncut circuit
        uncut_vqe_circuit_expval = vqe_circuit_qnode(num_qubits, vqe_params, ricco_params, 
                                                     obs, unitary_rotation=False)

        # Convert the uncut QNode to a quantum tape for processing
        tape = vqe_circuit_qnode.qtape

        # Perform circuit cutting on the quantum tape
        graph = qml.transforms.qcut.tape_to_graph(tape)
        qml.transforms.qcut.replace_wire_cut_nodes(graph)

        # Generate fragments and the communication graph for circuit cutting
        # fragments, communication_graph = qml.transforms.qcut.fragment_graph(graph)
        fragments, communication_graph = ricco_fragment_graph(graph)
        fragment_tapes = [qml.transforms.qcut.graph_to_tape(f) for f in fragments]

        # Identify the subcircuit that contains MeasureNodes
        subcircuit_with_measure = None
        for subcircuit in fragment_tapes:
            if any(isinstance(op, qml.transforms.qcut.MeasureNode) for op in subcircuit.operations):
                subcircuit_with_measure = subcircuit
                break

        # Raise an error if no subcircuit with MeasureNodes is found
        if subcircuit_with_measure is None:
            raise ValueError("No subcircuit contains MeasureNodes.")
         
        # This is just a dummy for future use
        # Identify the other subcircuit (assumes exactly two subcircuits)
        downstream_subcircuit = fragment_tapes[1] if fragment_tapes[0] is subcircuit_with_measure else fragment_tapes[0]


        # Extract the observables from subcircuit_with_measure
        ob = subcircuit_with_measure.observables
        observable_list.append(ob[0])

    # Create a new tape from subcircuit_with_measure with updated observables
    with qml.tape.QuantumTape() as upstream_subcircuit:
        # Reapply operations from the original subcircuit
        for op in subcircuit_with_measure.operations:
            op.queue()

        # Replace observables with new observables from observable_list
        for obs in observable_list:
            qml.expval(obs)

    # new_tape now has the operations from subcircuit_with_measure and observables from observable_list

    return upstream_subcircuit, downstream_subcircuit