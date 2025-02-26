# RICCO: Rotation-Inspired Circuit Cut Optimization

## Overview

This repository contains the implementation of the **Rotation-Inspired Circuit Cut Optimization (RICCO)** algorithm, as described in the paper:

**Title**: Rotation-Inspired Circuit Cut Optimization  
**Authors**: Gideon Uchehara, Tor M. Aamodt, Olivia Di Matteo  
**Conference**: IEEE International Conference on Quantum Computing and Engineering (QCE) 2022  
**Paper Link**: [IEEE Xplore](https://www.computer.org/csdl/proceedings-article/qcs/2022/753600a050/1KnWBzpcrHq)  
**arXiv Version**: [arXiv:2211.07358](https://arxiv.org/abs/2211.07358)  

**RICCO** is a novel method for reducing the post-processing overhead of quantum circuit cutting by introducing parameterized unitary rotations at cut locations. This approach reduces the number of quantum circuit executions required to reconstruct the output of a large quantum circuit, making it more efficient for execution on small-scale quantum devices.

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Algorithm Overview](#algorithm-overview)
- [Examples](#examples)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)
- [Citation](#citation)

## Introduction

Quantum circuit cutting is a technique that allows large quantum circuits to be divided into smaller subcircuits, which can be executed on smaller quantum devices. However, classical post-processing to reconstruct the original circuit's output is computationally expensive, scaling exponentially with the number of cuts. **RICCO** addresses this issue by introducing **parameterized unitary rotations** at cut locations, optimizing these rotations to align the quantum state with specific observables, thereby reducing the number of measurements required.

This repository provides the implementation of **RICCO**, along with examples and tools to simulate and analyze its performance.

## Installation

To use **RICCO**, you need to install the required dependencies. The code is implemented in **Python** and uses the **PennyLane** library for quantum circuit simulation and optimization.

### Prerequisites

- Python 3.7 or higher
- pip (Python package manager)

### Installation Steps

1. Clone the repository:

   ```bash
   git clone https://github.com/gideonuchehara/RICCO-Circuit-Cutting-Algorithm.git
   cd RICCO-Circuit-Cutting-Algorithm
   ```

2. Install the required Python packages:

   ```bash
   pip install -r requirements.txt
   ```

3. Verify the installation by running a simple test script:

   ```bash
   python test_installation.py
   ```

## Usage

The **RICCO** algorithm is implemented in the `ricco.py` file. You can use it to optimize quantum circuit cuts and reconstruct the output of large quantum circuits.

### Basic Usage

#### Define Your Quantum Circuit

Use **PennyLane** to define your quantum circuit. For example:

```python
import pennylane as qml

dev = qml.device("default.qubit", wires=4)

@qml.qnode(dev)
def circuit(params):
    qml.RX(params[0], wires=0)
    qml.RY(params[1], wires=1)
    qml.CNOT(wires=[0, 1])
    return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))
```

#### Initialize RICCO

Create an instance of the **RICCO** class with your circuit, number of cuts, and device:

```python
from ricco import RICCO

num_cuts = 1
ricco = RICCO(circuit, num_cuts, dev)
```

#### Run the Optimization

Use the `reconstruct` method to optimize the circuit cuts and reconstruct the output:

```python
expectation_value = ricco.reconstruct()
print("Expectation Value:", expectation_value)
```

#### Run Multiple Trials

To reduce randomness and improve accuracy, you can run multiple trials and compute the mean expectation value:

```python
mean_expval = ricco.run_multiple_trials(iterations=5)
print("Mean Expectation Value:", mean_expval)
```

## Algorithm Overview

The **RICCO** algorithm works as follows:

1. **Circuit Cutting**:
   - The quantum circuit is divided into smaller subcircuits at specified cut locations.
   - Each cut introduces a **MeasureNode** and a **PrepareNode**.

2. **Unitary Rotation**:
   - A parameterized **unitary rotation** is applied at each cut location.
   - The rotation is optimized to align the quantum state with specific observables (e.g., Pauli-Z).

3. **Optimization**:
   - The cost function is minimized to ensure that the expectation values of certain observables are maximized, while others are set to zero.
   - This reduces the number of measurements required.

4. **Reconstruction**:
   - The results from the subcircuits are combined using classical post-processing to reconstruct the output of the original circuit.

For more details, refer to the **paper**.

## Examples

### Example 1: Simple Circuit with 1 Cut

```python
import pennylane as qml
from ricco import RICCO

# Define a simple quantum circuit
dev = qml.device("default.qubit", wires=2)

@qml.qnode(dev)
def circuit(params):
    qml.RX(params[0], wires=0)
    qml.RY(params[1], wires=1)
    qml.CNOT(wires=[0, 1])
    return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

# Initialize RICCO
num_cuts = 1
ricco = RICCO(circuit, num_cuts, dev)

# Reconstruct the expectation value
expectation_value = ricco.reconstruct()
print("Expectation Value:", expectation_value)
```

### Example 2: Running Multiple Trials

```python
# Run multiple trials to reduce randomness
mean_expval = ricco.run_multiple_trials(iterations=5)
print("Mean Expectation Value:", mean_expval)
```

## Results

The repository includes scripts to reproduce the results from the paper. You can find the following:

- **Accuracy Comparison**: Compare the accuracy of RICCO with other circuit-cutting methods.
- **Execution Count**: Analyze the number of quantum circuit executions required for RICCO and other methods.
- **VQE Application**: Apply RICCO to the Variational Quantum Eigensolver (VQE) for molecular ground state energy estimation.

To run the experiments, use the provided scripts in the `experiments/` directory.

## Contributing

Contributions to this repository are welcome! If you find any issues or have suggestions for improvements, please **open an issue** or **submit a pull request**.

1. Fork the repository.
2. Create a new branch for your feature or bugfix.
3. Commit your changes and push to the branch.
4. Submit a pull request.

## License

This project is licensed under the **MIT License**. See the `LICENSE` file for details.

## Citation

If you use **RICCO** in your research, please cite the following paper:

```bibtex
@inproceedings{ricco2022,
  title={Rotation-Inspired Circuit Cut Optimization},
  author={Uchehara, Gideon and Aamodt, Tor M and Di Matteo, Olivia},
  booktitle={2022 IEEE International Conference on Quantum Computing and Engineering (QCE)},
  pages={505--515},
  year={2022},
  organization={IEEE}
}
```

For any questions or feedback, please contact **Gideon Uchehara**.

