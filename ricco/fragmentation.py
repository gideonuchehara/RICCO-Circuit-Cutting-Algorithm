import pennylane as qml
from pennylane import numpy as np
from pennylane.transforms.qcut import MeasureNode, PrepareNode
from itertools import product
from typing import List, Tuple, Sequence
import string
from networkx import MultiDiGraph


"""
Processing functions for RICCO circuit cutting, adapted from PennyLane's implementation.
This module defines functions used in the RICCO framework for reconstructing the output
of an uncut quantum circuit from its fragments.
"""

def ricco_processing_fn(
    results: Sequence[Sequence],
    communication_graph: MultiDiGraph,
    prepare_nodes: Sequence[Sequence[PrepareNode]],
    measure_nodes: Sequence[Sequence[MeasureNode]],
    use_opt_einsum: bool = False,
):
    """
    Processes results of circuit fragments in a modified RICCO circuit cutting workflow
    by constructing tensors and contracting them according to the communication graph.
    
    This function modifies the :func:`cut_circuit()` transform for the RICCO framework, 
    reconstructing the outcome of the original uncut circuit.

    Args:
        results (Sequence[Sequence]): A collection of execution results from expanding circuit
            fragments over measurement and preparation node configurations. These results are 
            processed into tensors and then contracted.
        communication_graph (MultiDiGraph): The communication graph that defines connectivity 
            between circuit fragments.
        prepare_nodes (Sequence[Sequence[PrepareNode]]): Sequence where each entry corresponds 
            to preparation nodes in each fragment. Defines the order of preparation indices for 
            tensor construction.
        measure_nodes (Sequence[Sequence[MeasureNode]]): Sequence where each entry corresponds 
            to measurement nodes in each fragment. Defines the order of measurement indices for 
            tensor construction.
        use_opt_einsum (bool): If True, uses the `opt_einsum` package for optimized tensor 
            contraction. `opt_einsum` provides faster contraction for large networks but requires 
            installation via `pip install opt_einsum`. Defaults to False.

    Returns:
        float or tensor_like: The reconstructed output of the original uncut circuit, obtained 
        from contracting the tensor network of circuit fragments.

    Notes:
        This function is specifically designed for use in the RICCO circuit cutting workflow.
        For details on the general circuit cutting process, refer to the 
        :func:`qml.cut_circuit()` transform.
    """
    
    # Ensure compatibility with active return types
    if qml.active_return():
        results = [
            qml.math.stack(tape_res) if isinstance(tape_res, tuple) 
            else qml.math.reshape(tape_res, [-1])
            for tape_res in results
        ]

    # Flatten the results from each circuit fragment into a single array
    flat_results = qml.math.concatenate(results)

    # Convert results into tensors based on prepare and measure nodes
    tensors = _to_tensors(flat_results, prepare_nodes, measure_nodes)

    # Perform tensor contraction based on the communication graph and node order
    result = contract_tensors(
        tensors, communication_graph, prepare_nodes, measure_nodes, use_opt_einsum
    )

    return result






def _reshape_results(results: Sequence, shots: int) -> List[List]:
    """
    Reshapes ``results`` into a two-dimensional nested list, where the number of rows
    corresponds to the number of cuts, and the number of columns corresponds to the 
    number of shots for each fragment.

    This helper function processes measurement results from circuit fragments, making
    them compatible for tensor operations in the RICCO framework.

    Args:
        results (Sequence): A collection of results from executing circuit fragments, 
            containing either expectation values or sample measurements.
        shots (int): Number of shots for each fragment, determining the column count 
            of the reshaped output.

    Returns:
        List[List]: A two-dimensional nested list with rows corresponding to circuit 
        fragments and columns corresponding to shots.
    """
    
    if qml.active_return():
        # Ensure results have compatible shapes by stacking, avoiding ragged arrays
        results = [
            qml.math.stack(tape_res) if isinstance(tape_res, tuple) else tape_res
            for tape_res in results
        ]

    # Flatten each result entry for consistent handling in tensor operations
    results = [qml.math.flatten(r) for r in results]

    # Group results into chunks based on shot count
    results = [results[i : i + shots] for i in range(0, len(results), shots)]

    # Transpose the results list to ensure rows represent fragments and columns represent shots
    results = list(map(list, zip(*results)))

    return results




def _get_symbol(i: int) -> str:
    """
    Returns the i-th ASCII letter symbol, supporting both lowercase and uppercase letters.
    Allows values of `i` up to 51, corresponding to 26 lowercase and 26 uppercase letters.

    Args:
        i (int): Index of the desired ASCII letter symbol.

    Returns:
        str: The ASCII letter at position `i`.

    Raises:
        ValueError: If `i` exceeds the range of available ASCII letters (up to 51).
    """
    if i >= len(string.ascii_letters):
        raise ValueError(
            "Exceeded available ASCII symbols. Set `use_opt_einsum` to True when applying "
            f"more than {len(string.ascii_letters)} wire cuts to a circuit."
        )
    
    return string.ascii_letters[i]





def contract_tensors(
    tensors: Sequence,
    communication_graph: MultiDiGraph,
    prepare_nodes: Sequence[Sequence[PrepareNode]],
    measure_nodes: Sequence[Sequence[MeasureNode]],
    use_opt_einsum: bool = False,
):
    """
    Contracts tensors according to the edges specified in the communication graph.

    This function is used within the RICCO circuit cutting framework to perform tensor
    contractions based on a directed communication graph. The graph defines connections
    between tensors and represents each fragment's preparation and measurement indices.

    Args:
        tensors (Sequence): The tensors to be contracted, each representing a circuit fragment.
        communication_graph (MultiDiGraph): A directed multigraph defining connectivity between 
            tensors, where each edge indicates a contraction.
        prepare_nodes (Sequence[Sequence[PrepareNode]]): Order of preparation indices for each 
            tensor, used to map back to the contraction equation.
        measure_nodes (Sequence[Sequence[MeasureNode]]): Order of measurement indices for each 
            tensor, used to map back to the contraction equation.
        use_opt_einsum (bool): If True, uses `opt_einsum` for optimized tensor contraction.
            This can provide faster contraction for large networks but requires separate 
            installation (`pip install opt_einsum`).

    Returns:
        float or tensor_like: The result of contracting the tensor network.

    Raises:
        ImportError: If `use_opt_einsum` is True but `opt_einsum` is not installed.

    Notes:
        This function is designed to work with the RICCO circuit cutting workflow. Each tensor 
        is assigned a unique symbol to represent indices, and the communication graph guides 
        the tensor contraction based on edges defined by `PrepareNode` and `MeasureNode` pairs.
    """
    # Import opt_einsum if specified for optimized contraction; otherwise use standard einsum
    if use_opt_einsum:
        try:
            from opt_einsum import contract, get_symbol
        except ImportError as e:
            raise ImportError(
                "The opt_einsum package is required when `use_opt_einsum` is set to True in "
                "the `contract_tensors` function. Install it using:\n\npip install opt_einsum"
            ) from e
    else:
        contract = qml.math.einsum
        get_symbol = _get_symbol

    # Initialize the symbol counter and storage for tensor indices
    ctr = 0
    tensor_indices = [""] * len(communication_graph.nodes)
    measure_to_symbol_map = {}

    # Assign symbols to preparation indices based on the communication graph
    for i, (node, prep_nodes) in enumerate(zip(communication_graph.nodes, prepare_nodes)):
        predecessors = communication_graph.pred[node]
        
        for prep_node in prep_nodes:
            for _, pred_edges in predecessors.items():
                for pred_edge in pred_edges.values():
                    meas_op, prep_op = pred_edge["pair"]

                    # Assign a unique symbol to the preparation-measurement pair
                    if prep_node.id == prep_op.id:
                        symbol = get_symbol(ctr)
                        ctr += 1
                        tensor_indices[i] += symbol
                        measure_to_symbol_map[meas_op] = symbol

    # Append symbols for measurement indices based on successors in the communication graph
    for i, (node, meas_nodes) in enumerate(zip(communication_graph.nodes, measure_nodes)):
        successors = communication_graph.succ[node]
        
        for meas_node in meas_nodes:
            for _, succ_edges in successors.items():
                for succ_edge in succ_edges.values():
                    meas_op, _ = succ_edge["pair"]

                    # Retrieve the symbol for the matching measurement operation
                    if meas_node.id == meas_op.id:
                        symbol = measure_to_symbol_map[meas_op]
                        tensor_indices[i] += symbol

    # Build the contraction equation from tensor indices
    contraction_equation = ",".join(tensor_indices)
    einsum_kwargs = {} if use_opt_einsum else {"like": tensors[0]}

    # Perform the tensor contraction and return the result
    return contract(contraction_equation, *tensors, **einsum_kwargs)





CHANGE_OF_BASIS = qml.math.array([[1.0, 1.0], [1.0, -1.0]])

def _process_tensor(results, n_prep: int, n_meas: int):
    """
    Converts a flat slice of an individual circuit fragment's execution results into a tensor
    representation for RICCO circuit cutting.

    This function performs the following steps:

    1. Reshapes `results` into the intermediate shape `(2,) * n_prep + (2**n_meas,)`.
    2. Reorders the final axis to follow the standard product order over measurement settings.
       For `n_meas = 2`, the standard product order is: II, IZ, ZI, ZZ, while the input order
       corresponds to `qml.pauli.partition_pauli_group(2)`, i.e., II, IZ, ZI, ZZ.
    3. Reshapes into the final target shape `(2,) * (n_prep + n_meas)`.
    4. Applies a change of basis for the preparation indices (first `n_prep` indices) from the
       |0>, |1> basis to the I, Z basis using `CHANGE_OF_BASIS`.

    Args:
        results (tensor_like): The input execution results for the circuit fragment.
        n_prep (int): Number of preparation nodes in the circuit fragment.
        n_meas (int): Number of measurement nodes in the circuit fragment.

    Returns:
        tensor_like: The processed tensor representing the circuit fragment.
    """
    
    # Total number of indices for the tensor (preparation + measurement)
    n = n_prep + n_meas
    dim_meas = 2 ** n_meas  # Dimensionality for measurement settings

    # Step 1: Reshape results into the intermediate shape based on prep and measurement counts
    intermediate_shape = (2,) * n_prep + (dim_meas,)
    intermediate_tensor = qml.math.reshape(results, intermediate_shape)

    # Step 2: Reorder measurement configurations to standard product order
    grouped = qml.pauli.partition_pauli_group(n_meas)[0]
    grouped_flat = [term for term in grouped]  # Flatten the grouping
    order = qml.math.argsort(grouped_flat)

    # Adjust reshaping for TensorFlow, which lacks slicing support
    if qml.math.get_interface(intermediate_tensor) == "tensorflow":
        intermediate_tensor = qml.math.gather(intermediate_tensor, order, axis=-1)
    else:
        slice_order = [slice(None)] * n_prep + [order]
        intermediate_tensor = intermediate_tensor[tuple(slice_order)]

    # Step 3: Reshape to final shape `(2,) * (n_prep + n_meas)`
    final_shape = (2,) * n
    final_tensor = qml.math.reshape(intermediate_tensor, final_shape)

    # Step 4: Apply the change of basis for preparation indices using `CHANGE_OF_BASIS`
    change_of_basis = qml.math.convert_like(CHANGE_OF_BASIS, intermediate_tensor)
    for i in range(n_prep):
        final_tensor = qml.math.tensordot(change_of_basis, final_tensor, axes=[[1], [i]])

    # Reorder indices due to `tensordot` output order
    axis_order = list(reversed(range(n_prep))) + list(range(n_prep, n))
    final_tensor = qml.math.transpose(final_tensor, axes=axis_order)

    # Normalize the final tensor by a factor of `2^(-(n_meas + n_prep) / 2)`
    final_tensor *= qml.math.power(2, -(n_meas + n_prep) / 2)

    return final_tensor



def _to_tensors(
    results,
    prepare_nodes: Sequence[Sequence[PrepareNode]],
    measure_nodes: Sequence[Sequence[MeasureNode]],
) -> List:
    """
    Processes a flat list of execution results from circuit fragments into a list of tensors,
    where each tensor corresponds to a circuit fragment.

    This function slices `results` based on the expected size of each fragment tensor, determined
    by the number of preparation and measurement nodes. The sliced results are then processed by
    `_process_tensor` for further transformation.

    Args:
        results (tensor_like): A flat tensor of execution results corresponding to the expansion
            of circuit fragments over measurement and preparation node configurations.
        prepare_nodes (Sequence[Sequence[PrepareNode]]): A sequence where each entry contains the
            preparation nodes for a fragment, defining the number of preparation nodes.
        measure_nodes (Sequence[Sequence[MeasureNode]]): A sequence where each entry contains the
            measurement nodes for a fragment, defining the number of measurement nodes.

    Returns:
        List[tensor_like]: A list of tensors, where each tensor represents a circuit fragment
        in the communication graph.

    Raises:
        ValueError: If the length of `results` does not match the expected size based on the 
        number of preparation and measurement nodes.
    """
    
    ctr = 0  # Counter to track position within the flat `results`
    tensors = []

    # Process each fragment's results based on its number of preparation and measurement nodes
    for prep, meas in zip(prepare_nodes, measure_nodes):
        n_prep = len(prep)  # Number of preparation nodes in the fragment
        n_meas = len(meas)  # Number of measurement nodes in the fragment
        total_nodes = n_prep + n_meas

        # Determine the slice size for the fragment's results
        dim = 2 ** total_nodes
        results_slice = results[ctr: ctr + dim]

        # Convert the sliced results into a tensor for the fragment
        tensors.append(_process_tensor(results_slice, n_prep, n_meas))

        # Update the counter to the next position in the flat `results`
        ctr += dim

    # Validate that the total processed results match the expected size
    if results.shape[0] != ctr:
        raise ValueError(f"The `results` argument should have a length of {ctr} to match the expected size.")

    return tensors
