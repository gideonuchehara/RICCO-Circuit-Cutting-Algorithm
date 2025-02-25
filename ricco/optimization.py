import pennylane as qml
from pennylane import numpy as np
from typing import List, Tuple
from .utils import replace_measure_with_unitary, generate_n_qubit_unitary



def ricco_cost_function(fragment_tapes, params: np.ndarray) -> float:
    """
    Computes the Ricco cost function for the current unitary parameters applied to cut qubits.
    
    Args:
        fragment_tapes (List[QuantumScript]): List of fragment tapes containing subcircuits of the main circuit.
        params (np.ndarray): Current parameters for the unitary rotation.

    Returns:
        float: The computed Ricco cost based on the squared deviation from ideal expectations.
    """
    # Create an upstream subcircuit with MeasureNodes replaced by unitary rotations
    upstream_subcircuit = replace_measure_with_unitary(fragment_tapes, params, generate_n_qubit_unitary)

    # dev = qml.device("default.qubit", wires=len(upstream_subcircuit.wires))
    dev = qml.device("default.qubit", wires=upstream_subcircuit.wires)
    
    # Execute the modified subcircuit and retrieve expectation values
    expvals = qml.execute([upstream_subcircuit], dev, qml.gradients.param_shift)[0]

    # Calculate the cost based on squared deviations from the ideal outcome (all ones)
    ones_matrix = np.ones_like(expvals)
    expvals_squared = np.square(expvals)
    F = ones_matrix - expvals_squared
    return np.sum(F)

# def ricco_cost_function(upstream_subcircuit, params: np.ndarray) -> float:
#     """
#     Computes the Ricco cost function for the current unitary parameters applied to cut qubits.
    
#     Args:
#         fragment_tapes (List[QuantumScript]): List of fragment tapes containing subcircuits of the main circuit.
#         params (np.ndarray): Current parameters for the unitary rotation.

#     Returns:
#         float: The computed Ricco cost based on the squared deviation from ideal expectations.
#     """
#     # # Create an upstream subcircuit with MeasureNodes replaced by unitary rotations
#     # upstream_subcircuit = replace_measure_with_unitary(fragment_tapes, params, generate_n_qubit_unitary)

#     # dev = qml.device("default.qubit", wires=len(upstream_subcircuit.wires))
#     dev = qml.device("default.qubit", wires=upstream_subcircuit.wires)
    
#     # Execute the modified subcircuit and retrieve expectation values
#     expvals = qml.execute([upstream_subcircuit], dev, qml.gradients.param_shift)[0]

#     # Calculate the cost based on squared deviations from the ideal outcome (all ones)
#     ones_matrix = np.ones_like(expvals)
#     expvals_squared = np.square(expvals)
#     F = ones_matrix - expvals_squared
#     return np.sum(F)



def ricco_unitary_optimization(fragment_tapes: List[qml.tape.QuantumScript], params: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray], List[float]]:
    """
    Optimizes the unitary rotation parameters applied to cut qubits in a fragment of a quantum circuit
    using a custom Ricco cost function. The cost function minimizes the difference between 
    expected and ideal measurement outcomes for the cut qubits.

    Args:
        fragment_tapes (List[QuantumScript]): List of fragment tapes containing subcircuits of the main circuit.
        params (np.ndarray): Initial parameters for the unitary rotation on the cut qubits.

    Returns:
        Tuple:
            - np.ndarray: Final optimized parameters.
            - List[np.ndarray]: History of parameter updates throughout the optimization process.
            - List[float]: History of cost values for each optimization step.

    Steps:
    1. Define a Ricco cost function based on the difference between expected and ideal measurement outcomes.
    2. Use the Adam optimizer to iteratively update parameters until convergence.

    **Example**

    .. code-block:: python

        fragment_tapes = [<QuantumScript object>, <QuantumScript object>]  # obtained from circuit cutting
        initial_params = np.random.rand(10)  # random initial parameters
        optimized_params, param_history, cost_history = ricco_unitary_optimization(fragment_tapes, initial_params)
    """

    
    # Optimization setup
    optimal_params = []
    N_steps = 0
    conv_tol = 1e-6  # Convergence tolerance for stopping criterion
    step_size = 0.5
    tol = 10
    ricco_opt = qml.AdamOptimizer(stepsize=step_size)
    prev_cost = 100
    cost_history = [prev_cost]

    # Optimization loop
    while tol > conv_tol:
        # Perform an optimization step and compute the new cost
        (_, params), new_cost = ricco_opt.step_and_cost(ricco_cost_function, fragment_tapes, params)

        # Update convergence criteria and log values
        tol = np.abs(prev_cost - new_cost)
        prev_cost = new_cost
        cost_history.append(prev_cost)
        optimal_params.append(params)
        N_steps += 1

    return params, optimal_params, cost_history

# def ricco_unitary_optimization(upstream_subcircuit: List[qml.tape.QuantumScript], params: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray], List[float]]:
#     """
#     Optimizes the unitary rotation parameters applied to cut qubits in a fragment of a quantum circuit
#     using a custom Ricco cost function. The cost function minimizes the difference between 
#     expected and ideal measurement outcomes for the cut qubits.

#     Args:
#         fragment_tapes (List[QuantumScript]): List of fragment tapes containing subcircuits of the main circuit.
#         params (np.ndarray): Initial parameters for the unitary rotation on the cut qubits.

#     Returns:
#         Tuple:
#             - np.ndarray: Final optimized parameters.
#             - List[np.ndarray]: History of parameter updates throughout the optimization process.
#             - List[float]: History of cost values for each optimization step.

#     Steps:
#     1. Define a Ricco cost function based on the difference between expected and ideal measurement outcomes.
#     2. Use the Adam optimizer to iteratively update parameters until convergence.

#     **Example**

#     .. code-block:: python

#         fragment_tapes = [<QuantumScript object>, <QuantumScript object>]  # obtained from circuit cutting
#         initial_params = np.random.rand(10)  # random initial parameters
#         optimized_params, param_history, cost_history = ricco_unitary_optimization(fragment_tapes, initial_params)
#     """

    
#     # Optimization setup
#     optimal_params = []
#     N_steps = 0
#     conv_tol = 1e-6  # Convergence tolerance for stopping criterion
#     step_size = 0.5
#     tol = 10
#     ricco_opt = qml.AdamOptimizer(stepsize=step_size)
#     prev_cost = 100
#     cost_history = [prev_cost]

#     # Optimization loop
#     while tol > conv_tol:
#         # Perform an optimization step and compute the new cost
#         (_, params), new_cost = ricco_opt.step_and_cost(ricco_cost_function, upstream_subcircuit, params)

#         # Update convergence criteria and log values
#         tol = np.abs(prev_cost - new_cost)
#         prev_cost = new_cost
#         cost_history.append(prev_cost)
#         optimal_params.append(params)
#         N_steps += 1

#     return params, optimal_params, cost_history