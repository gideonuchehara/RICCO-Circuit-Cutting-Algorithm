from scipy.stats import unitary_group
import pennylane as qml

def generate_random_circuit(num_qubits, num_cuts=1, 
                            seed_u=0, seed_v=0):
    """
    Generates a random unitary quantum circuit with specified qubits and cuts.
    
    Parameters:
    - num_qubits (int): Total number of qubits in the circuit.
    - num_cuts (int): Number of cut qubits between subcircuits.
    - seed_u (int): Seed for generating the random unitary in subcircuit AB.
    - seed_v (int): Seed for generating the random unitary in subcircuit BC.
    
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
    # observable = qml.grouping.string_to_pauli_word("Z" * num_qubits)
    observable =  qml.pauli.string_to_pauli_word("Z" * num_qubits)
    
    # Generate random unitaries for subcircuits AB and BC
    U_ab = unitary_group.rvs(2 ** len(AB_wires), random_state=seed_u)
    U_bc = unitary_group.rvs(2 ** len(BC_wires), random_state=seed_v)
    
    # Apply unitaries and wire cut
    qml.QubitUnitary(U_ab, wires=range(len(AB_wires)))
    
    qml.WireCut(wires=range(N_A_wires, N_A_wires + N_B_wires))
    
    qml.QubitUnitary(U_bc, wires=range(N_A_wires, num_qubits))
    
    # Return expectation value of the observable
    return qml.expval(observable)