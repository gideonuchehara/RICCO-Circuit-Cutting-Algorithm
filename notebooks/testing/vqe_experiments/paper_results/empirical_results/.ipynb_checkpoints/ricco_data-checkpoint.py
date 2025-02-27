#!/usr/bin/env python
# coding: utf-8

import pennylane as qml
from pennylane import numpy as np
from scipy.stats import unitary_group
from scipy.stats import ortho_group
import matplotlib.pyplot as plt

import random
from collections import Counter, OrderedDict
from itertools import combinations
from itertools import compress, product


def ricco(N_A_wires, N_B_wires, N_C_wires, group, tolerance=1e-12):
    """
    This is the RICCO function to collect data for comparison with
    QCUT.
    INPUTS:
        N_A_wires: total number of qubits for wires labeled A
        N_B_wires: total number of qubits for wires labeled B. 
                   This is the optimization qubits. either 1 
                   or 2 qubits
        N_C_wires: total number of qubits for wires labeled C
        tolerance: Tolerance for the the optmization. set to 1e-12
        group    : The type of unitary operator used for circuit.
                   Either "unitary" or "ortho"
    OUTPUTS:
        ricco_dict: Dictionary of RICCO results
        qcut_dict : Dictionary of QCUT results
        
    """
    
    ###################### Global Variables

    N_AB_wires = N_A_wires + N_B_wires            # subcircuit_AB total wires
    N_BC_wires = N_B_wires + N_C_wires            # subcircuit_BC total wires
    N_wires = N_A_wires + N_B_wires + N_C_wires   # total wires in circuit

    # Generating labels for the differenct wires
    A_wires = ["A" + str(x) for x in list(range(N_A_wires))]
    B_wires = ["B" + str(x) for x in list(range(N_B_wires))]
    C_wires = ["C" + str(x) for x in list(range(N_C_wires))]

    # mapping the wires to their respective labels
    A_wire_map = {x: idx for idx, x in enumerate(A_wires)}
    B_wire_map = {x: idx for idx, x in enumerate(B_wires)}
    C_wire_map = {x: idx for idx, x in enumerate(C_wires)}

    # wire labels for the different subcircuits
    AB_wires = A_wires + B_wires              # subcircuit_AB wire labels
    BC_wires = B_wires + C_wires              # subcircuit_BC twire labels
    all_wires = A_wires + B_wires + C_wires   # total wire labels

    # wire mapping for the different subcircuits
    AB_wires_map = {x: idx for idx, x in enumerate(AB_wires)}
    BC_wires_map = {x: idx for idx, x in enumerate(BC_wires)}
    all_wires_map = {x: idx for idx, x in enumerate(all_wires)}

    # Generating and ordering Pauli group for N_B_wires
    N_B_grouped = qml.grouping.partition_pauli_group(N_B_wires)
    N_B_grouped_flat = [term for group in N_B_grouped for term in group]
    N_B_order = qml.math.argsort(N_B_grouped_flat)

    # Measurement observables for the different wire labels
    N_A_pauli_word = "Z" * (N_A_wires)
    N_B_pauli_word = qml.math.array(N_B_grouped_flat)[N_B_order]
    N_BC_pauli_word = "Z" * (N_B_wires + N_C_wires)

    # Main observable for the entire circuit
    obs = "Z" * N_wires
    op = qml.grouping.string_to_pauli_word(obs, wire_map=all_wires_map)

    # Pauli strings for the different subcircuits
    AB_pauli_labels = [N_A_pauli_word + word for word in N_B_pauli_word]
    BC_pauli_labels = [N_BC_pauli_word]

    # convert the Pauli strings to Pauli observables mappped to the respective wires
    AB_paulis = [qml.grouping.string_to_pauli_word(P, wire_map=AB_wires_map) 
                  for P in AB_pauli_labels]
    BC_paulis = [qml.grouping.string_to_pauli_word(P, wire_map=BC_wires_map) 
                  for P in BC_pauli_labels]

    # Paulis with only Z or I
    opt_pauli_string = [P for P in N_B_pauli_word if all( [c in set('IZ') for c in P])]
    opt_paulis = [qml.grouping.string_to_pauli_word(P, wire_map=AB_wires_map) 
                  for P in AB_pauli_labels if all( [c in set('IZ') for c in P]
    )]


    # Pauli I and Z and their respective eigenvectors and eigenvalues
    I = ["0", "1"]                   # eigenvectors of I
    I_sign = qml.math.array([1, 1])  # eigenvalues of I

    Z = ["0", "1"]                   # eigenvectors of Z
    Z_sign = qml.math.array([1, -1]) # eigenvalues of Z

    meas_to_prep_state_dic = {"I":I, "Z":Z}           # Dictionary for eigenvectors
    meas_to_prep_sign_dic = {"I":I_sign, "Z":Z_sign}  # Dictionary for eigenvalues

    eig_list = []
    for string in opt_pauli_string:
        eig = 0
        for indx, ob in enumerate(string):
            if indx==0:
                eig = meas_to_prep_sign_dic[ob]
            else:
                eig = qml.math.kron(eig, meas_to_prep_sign_dic[ob])
        eig_list.append(list(eig))

    eigvals = np.array(eig_list)

    # Random number to generate random unitaries for subcircuit_AB and Subcircuit_BC
    random_number_AB = random.randint(0, 1000)
    random_number_BC = random.randint(0, 1000)

    # Defining the Random unitaries for subcircuit_AB and subcircuit_BC
    if group == "unitary":
        U_ab = unitary_group.rvs(2 ** len(A_wires + B_wires), random_state=random_number_AB)
        U_bc = unitary_group.rvs(2 ** len(B_wires + C_wires), random_state=random_number_BC)
    elif group == "ortho":
        U_ab = ortho_group.rvs(2 ** len(A_wires + B_wires), random_state=random_number_AB)
        U_bc = ortho_group.rvs(2 ** len(B_wires + C_wires), random_state=random_number_BC)

    # Define Parameters
    params = np.random.uniform(-np.pi, np.pi, size=4**N_B_wires-1)


    
    # #### Original Uncut Circuit
    
    # Original uncut circuit with wircut at cut location
    dev = qml.device("default.qubit", wires=all_wires)
    @qml.qnode(dev)
    def uncut_circuit(U_ab, U_bc):
        qml.QubitUnitary(U_ab, wires=A_wires+B_wires)
        qml.WireCut(wires=B_wires)
        qml.QubitUnitary(U_bc, wires=B_wires+C_wires)
        return qml.expval(op)

    uncut_expval = float(uncut_circuit(U_ab, U_bc))



    # #### Unitary Rotation for cut location

    # Unitary operator to be used at cut location for one qubit cut
    def one_qubit_unitary(params, wires):
        """Ansatz for a general two-qubit unitary V."""
        qml.Rot(*params, wires=B_wires)

    # Conjugate of Unitary operator to be used at cut location for one qubit cut
    def one_qubit_unitary_dagger(params, wires):
        """Ansatz for a general two-qubit unitary V."""
        qml.adjoint(one_qubit_unitary)(params, wires)

    # Unitary operator to be used at cut location for two qubits cut
    def two_qubit_unitary(params, wires):
        """Ansatz for a general two-qubit unitary V."""
        qml.Rot(*params[0:3], wires=wires[0])
        qml.Rot(*params[3:6], wires=wires[1])
        qml.CNOT(wires=[wires[1], wires[0]])
        qml.RZ(params[6], wires=wires[0])
        qml.RY(params[7], wires=wires[1])
        qml.CNOT(wires=[wires[0], wires[1]])
        qml.RY(params[8], wires=wires[1])
        qml.CNOT(wires=[wires[1], wires[0]])
        qml.Rot(*params[9:12], wires=wires[0])
        qml.Rot(*params[12:], wires=wires[1])

    # Conjugate of Unitary operator to be used at cut location for two qubits cut
    def two_qubit_unitary_dagger(params, wires):
        """Ansatz for a general two-qubit unitary V."""
        qml.adjoint(two_qubit_unitary)(params, wires)

    # Quantum functions for subcircuit_AB
    def quantum_function_for_AB(U_ab, params):
        qml.QubitUnitary(U_ab, wires=A_wires+B_wires)
        if N_B_wires == 1:
            one_qubit_unitary(params, wires=B_wires)
        else:
            two_qubit_unitary(params, wires=B_wires)

    # Quantum functions for subcircuit_BC
    def quantum_function_for_BC(U_bc, params):
        if N_B_wires == 1:
            one_qubit_unitary_dagger(params, wires=B_wires)
        else:
            two_qubit_unitary_dagger(params, wires=B_wires)
        qml.QubitUnitary(U_bc, wires=B_wires+C_wires)


    # Defining the first half of the cut circuit - Subcircuit_AB
    dev = qml.device("default.qubit", wires=AB_wires)

    @qml.qnode(dev)
    def subcircuit_for_AB(U_ab, params):
        quantum_function_for_AB(U_ab, params)
        return [qml.expval(P) for P in opt_paulis]

    @qml.qnode(dev)
    def subcircuit_for_test(U_ab, params):
        quantum_function_for_AB(U_ab, params)
        return [qml.expval(P) for P in AB_paulis]

    
    # Defining the second half of the cut circuit - Subcircuit_AB
    dev = qml.device("default.qubit", wires=BC_wires)

    @qml.qnode(dev)
    def subcircuit_for_BC(U_bc, params):
        quantum_function_for_BC(U_bc, params)
        return [qml.expval(P) for P in BC_paulis]

    dev = qml.device("default.qubit", wires=all_wires)
    # Apply Qcut to Original uncut circuit
    @qml.cut_circuit()
    @qml.qnode(dev)
    def qcut_circuit(U_ab, U_bc):
        qml.QubitUnitary(U_ab, wires=A_wires+B_wires)
        qml.WireCut(wires=B_wires)
        qml.QubitUnitary(U_bc, wires=B_wires+C_wires)
        return qml.expval(op)

    with qml.Tracker(qcut_circuit.device) as qcut_exec_tracker:
        qcut_expval = qcut_circuit(U_ab, U_bc)

    qcut_exec = qcut_exec_tracker.totals["executions"]


    # ### Begin AdamOptimization for rotation parameters
    N_steps = 0

    # def cost_function(params):
    #     expvals = subcircuit_for_AB(U_ab, params)
    #     F = (1 - ((1/2**N_B_wires)**(2**N_B_wires))*np.sum(expvals * eigvals, axis=1))**2
    #     # F = (1 - np.sum(expvals * eigvals, axis=1))**2 # Gets stuck for some circuits
    #     return np.prod(F)
    # np.abs(arr)
    def cost_function(params):
        expvals = subcircuit_for_AB(U_ab, params)
        F = -np.abs(expvals)
        return F

    # Track the total number of executions for optimization
    with qml.Tracker(subcircuit_for_AB.device) as opt_tracker:
        """
        Track the total number of executions for optimization
            """
        tol = 10
        opt = qml.AdamOptimizer(stepsize=0.1)
        cost = []
        prev_cost = 100
        while tol > tolerance:
            params, new_cost = opt.step_and_cost(cost_function, params)

            # Calculate difference between new and old costs
            tol = np.abs((prev_cost - new_cost))
            
            prev_cost = new_cost
            
            N_steps += 1

    outputs = subcircuit_for_test(U_ab, params)
    optimizer_exec = opt_tracker.totals["executions"]


    # Compute the Deviation of the expectation values of zero-value observables from zero
    opt_keys = [qml.grouping.pauli_word_to_string(x, wire_map=AB_wires_map) for x in AB_paulis]
    opt_values = np.abs(outputs)
    opt_result = {key:float(value) for key, value in zip(opt_keys, opt_values)}

    zero_obs = [P for P in opt_keys if not all([c in set('IZ') for c in P])]
    zero_obs_expval = [opt_result[ob] for ob in zero_obs]
    total_zero_obs = len(zero_obs_expval)
    average_error = float(np.sum(np.array(zero_obs_expval))/total_zero_obs)


    # ### End of AdamOptimization
    
    

    # ###### Recontruction of Original Circuit's Expectation Value

    # Track the number of execution of subcircuit_AB after parameters are optimized
    with qml.Tracker(subcircuit_for_AB.device) as AB_exec_tracker:
        # Measurement of Subcircuit_AB
        meas = subcircuit_for_AB(U_ab, params)

        # Dictionary for Subcircuit_BC expectation values
        meas_labels = [word for word in AB_pauli_labels if all( [c in set('IZ') for c in word])]
        measure_result = {key: float(value) for key, value in zip(meas_labels, meas)}

    AB_exec = AB_exec_tracker.totals["executions"]
    

    # Subcircuit_BC redefined to account for initializations
    dev = qml.device("default.qubit", wires=BC_wires)

    # states combinations for observables I and Z for one and two qubits
    if N_B_wires == 1:
        states = Z
    else:
        states = [ m+n for m, n in list(product(I, Z))]

    @qml.qnode(dev)
    def subcircuit_for_BC_reconst(init, U_bc, params):
        for i, state in enumerate(init):
            if state == "1":
                qml.PauliX(wires=B_wires[i])

        quantum_function_for_BC(U_bc, params)
        return [qml.expval(P) for P in BC_paulis]

    # Track the number of execution of subcircuit_AB after parameters are optimized
    with qml.Tracker(subcircuit_for_BC_reconst.device) as BC_exec_tracker:
        # Measurement of Subcircuit_BC with initializations
        prep_measurements = []
        for init in states:
             prep_measurements.append(float(subcircuit_for_BC_reconst(init, 
                                                                      U_bc, params)[0]))
    BC_exec = BC_exec_tracker.totals["executions"]    

    # Dictionary for Subcircuit_BC expectation values
    prep = np.array(prep_measurements)
    prep_labels = states
    prepare_result = {key: float(value) for key, value in zip(prep_labels, prep)}
    

    def compute_expval(prepare_result):

        expvals = []
        for string in opt_pauli_string:
            for indx, ob in enumerate(string):
                if indx == 0:
                    x_state = meas_to_prep_state_dic[ob]
                    x_sign = meas_to_prep_sign_dic[ob]
                else:
                    y_state = meas_to_prep_state_dic[ob]
                    y_sign = meas_to_prep_sign_dic[ob]

                    x_state = [m+n for m, n in list(product(x_state, y_state))]
                    x_sign = qml.math.kron(x_sign, y_sign)

            p_word = N_A_pauli_word + string
            meas_expval = measure_result[p_word]

            prep_values = []
            for state in x_state:
                prep_values.append(prepare_result[state])

            prep_expval = qml.math.array(prep_values)

            combined_expval = meas_expval * prep_expval

            expvals.append(qml.math.dot(combined_expval, x_sign))


        total_expval = 1/(2**N_B_wires) * np.sum(qml.math.array(expvals))

        return total_expval


    ricco_expval = compute_expval(prepare_result)
    qcut_num_exec = qcut_execqcut_num_exec = qcut_exec
    ricco_num_exec = AB_exec + BC_exec
    ricco_expval_error = np.abs((uncut_expval - ricco_expval) * 100/uncut_expval)
    qcut_expval_error = np.abs((uncut_expval - qcut_expval) * 100/uncut_expval)


    ricco_dict ={ "Total number of qubits": N_wires,
                  "Number of cuts": N_B_wires,
                  "Number of executions": ricco_num_exec,
                  "Expectation error (%)": ricco_expval_error,
                  "Expectation value": ricco_expval,
                  "Method": "RICCO"}

    qcut_dict ={ "Total number of qubits": N_wires,
                 "Number of cuts": N_B_wires,
                 "Number of executions": qcut_num_exec,
                 "Expectation error (%)": qcut_expval_error,
                 "Expectation value": qcut_expval,
                 "Method": "QCUT"}
    
    opt_dict ={ "Number of iterations": N_steps,
                "Number of cuts": N_B_wires,
                "Number of executions": optimizer_exec,
                "Average error": average_error,}
    
    return ricco_dict, qcut_dict, opt_dict


