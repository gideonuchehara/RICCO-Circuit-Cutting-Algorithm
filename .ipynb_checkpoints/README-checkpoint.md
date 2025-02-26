# RICCO: Rotation-Inspired Circuit Cut Optimization

![RICCO Banner](https://github.com/gideonuchehara/RICCO-Circuit-Cutting-Algorithm/raw/main/assets/banner.png)

## 📌 Introduction
**RICCO (Rotation-Inspired Circuit Cut Optimization)** is an advanced quantum circuit-cutting algorithm designed to optimize circuit fragmentation by applying unitary transformations to MeasureNodes and updating their corresponding PrepareNodes. This method enhances post-processing efficiency and fidelity in circuit-cutting applications, allowing for scalable quantum computations.

This repository provides an implementation of the **RICCO** method as presented in the research paper:
> **"Rotation-Inspired Circuit Cut Optimization"** ([IEEE QCS 2022](https://www.computer.org/csdl/proceedings-article/qcs/2022/753600a050/1KnWBzpcrHq) | [arXiv version](https://arxiv.org/abs/2211.07358))

## 🚀 Features
✔️ **Automated Quantum Circuit Cutting**: Splits large quantum circuits into smaller subcircuits.  
✔️ **Optimized Unitary Transformations**: Applies Strongly Entangling Layers to MeasureNodes.  
✔️ **Parameter Optimization**: Uses gradient-based optimization to refine unitary parameters.  
✔️ **Seamless Integration**: Built on PennyLane, ensuring compatibility with various quantum backends.  
✔️ **Reconstruction of Quantum Circuits**: Efficient post-processing using tensor contraction.  

## 🛠 Installation
To install RICCO and its dependencies, follow these steps:

### 1️⃣ **Clone the Repository**
```bash
$ git clone https://github.com/gideonuchehara/RICCO-Circuit-Cutting-Algorithm.git
$ cd RICCO-Circuit-Cutting-Algorithm
```

### 2️⃣ **Create a Virtual Environment (Optional but Recommended)**
```bash
$ python -m venv ricco_env
$ source ricco_env/bin/activate  # On macOS/Linux
$ ricco_env\Scripts\activate    # On Windows
```

### 3️⃣ **Install Dependencies**
```bash
$ pip install -r requirements.txt
```

## 📜 Usage
### **Basic Example: Running RICCO on a Quantum Circuit**
```python
import pennylane as qml
from pennylane import numpy as np
from ricco.utils import generate_random_circuit

num_qubits = 5
num_cuts=1
seed_u = 103
seed_v = 105

# Initialize device for RICCO optimization
dev = qml.device("default.qubit", wires=range(num_qubits))

# Define and create a QNode for the generated quantum circuit
random_circuit_qnode = qml.QNode(generate_random_circuit, device=dev)

# Update QNode and Compute expectation value of the uncut circuit
uncut_random_circuit_expval = random_circuit_qnode(
    num_qubits, num_cuts, seed_u, seed_v)

# Optionally display the uncut circuit
fig1, ax = qml.draw_mpl(random_circuit_qnode)(num_qubits, num_cuts, 
                                              seed_u, seed_v)

tape = random_circuit_qnode.qtape


from ricco.algorithm import RICCO
# Initialize RICCO
ricco = RICCO(random_circuit_qnode, num_cuts, dev, entangling_layers=3)

# Run multiple trials and compute the average expectation value
ricco.run_multiple_trials(iterations=1)

# Access the mean expectation value
print("Mean Expectation Value (from attribute):", ricco.mean_expval)
```

### **Executing Optimization Process**
```python
ricco.optimize_unitary_rotations(steps=100, step_size=0.05, tol=1e-6)
print("Optimized Parameters:", ricco.params)
```

## 🔬 Methodology
### **Overview of Circuit Cutting**
Quantum circuits can be partitioned into fragments to reduce hardware constraints. **RICCO** improves upon traditional circuit-cutting methods by introducing **unitary rotations** at the cut locations, optimizing expectation values for post-processing efficiency.

### **Key Steps in RICCO**
1. **Circuit Partitioning**: Converts a quantum circuit into fragments using PennyLane's `qml.qcut`.
2. **Unitary Optimization**: Replaces MeasureNodes with unitary transformations (Strongly Entangling Layers) and optimizes parameters.
3. **Expectation Value Extraction**: Computes expectation values by executing the reconstructed subcircuits.
4. **Post-Processing with Tensor Contraction**: Uses an efficient post-processing function to reconstruct original circuit outputs.

For a detailed explanation, refer to the **[paper](https://arxiv.org/abs/2211.07358)**.

## 📖 Citation
If you use RICCO in your research, please cite:

```bibtex
@inproceedings{ricco2022,
  author    = {Gideon Uchehara and Others},
  title     = {Rotation-Inspired Circuit Cut Optimization},
  booktitle = {IEEE International Conference on Quantum Computing and Engineering (QCS)},
  year      = {2022},
  url       = {https://arxiv.org/abs/2211.07358}
}
```

## 🤝 Contributing
We welcome contributions! To contribute:
1. **Fork the repository**
2. **Create a new branch**
3. **Make your changes**
4. **Submit a pull request**

## 📄 License
This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

## 📬 Contact
For questions or collaborations, reach out to **Gideon Uchehara** via [GitHub](https://github.com/gideonuchehara).

---

⭐ If you find this project useful, please **star** this repository! ⭐

