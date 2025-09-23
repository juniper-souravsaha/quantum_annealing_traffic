# 🚦 Quantum Annealing for Traffic Optimization

This project explores **traffic congestion optimization** using both **classical simulated annealing (SA)** and **quantum-inspired QAOA (Quantum Approximate Optimization Algorithm)**.  
It demonstrates how road networks can be mapped to a **QUBO (Quadratic Unconstrained Binary Optimization)** problem and solved using different approaches.

---

## 📌 1. Introduction
Traffic congestion is a combinatorial optimization challenge.  
- **Classical methods** like Simulated Annealing are scalable but may get stuck in local minima.  
- **Quantum-inspired methods** like QAOA leverage quantum circuits to explore solution spaces differently.  

**Goal:** Compare **SA vs QAOA** on small traffic networks.

---

## 🏙️ 2. Problem Formulation
- Road network represented as a **graph**:
  - **Nodes** → intersections.  
  - **Edges** → roads (with congestion weights).  
- Optimization encoded as a **QUBO problem**.  
- **Objective:** minimize total congestion cost.

---

## ⚙️ 3. Methods

### 🔹 3.1 Classical Approach — Simulated Annealing (SA)
- Stochastic local search with **temperature schedule**.  
- Good for medium/large problem sizes.  

### 🔹 3.2 Quantum Approach — QAOA
- Implemented using **Qiskit** + **AerSimulator**.  
- Suitable for small QUBOs (≤ 20–30 variables).  
- Parameters: depth `reps`, optimizer (COBYLA/SPSA), measurement shots.  

---

## 🖥️ 4. Experiment Setup
- **Network sizes tested:** 6, 8, 10 nodes.  
- **SA parameters:** iterations, cooling schedule.  
- **QAOA parameters:** reps=1–3, optimizer=COBYLA, shots=128–512.  
- **Environment:**  
  - Python 3.12  
  - [NetworkX](https://networkx.org/)  
  - [Qiskit](https://qiskit.org/)  
  - (Optional) [D-Wave Ocean SDK](https://docs.ocean.dwavesys.com/)  

---

## 📊 5. Results
- **Comparison of SA vs QAOA:**  
  - Cost (objective value).  
  - Runtime (scalability).  
- **Example network visualization:**  
  - Nodes colored based on assignment (low vs high congestion).  

*(Plots generated in `demo_scalability.py`)*

---

## 💡 6. Key Insights
- **SA:** scalable, works well on larger graphs.  
- **QAOA:** promising for small graphs, highlights potential of quantum solvers.  
- **Tradeoff:** runtime grows rapidly for QAOA on simulators.  

---

## ✅ 7. Conclusion & Future Work
- Quantum approaches like QAOA can solve traffic optimization on small networks.  
- Scaling remains the key challenge.  
- **Future directions:**  
  - Run on **real quantum hardware** (IBM Q / D-Wave).  
  - Hybrid classical-quantum solvers.  
  - Larger city-scale datasets.  

---

## 🚀 8. How to Run

### Install dependencies
```bash
pip install -r requirements.txt
```

### Run comparison experiment
- compare_classical_quantum takes single grid_size as input to run.
```
python experiments/compare_classical_quantum.py
```

### Run scalability demo
- demo_scalability takes multiple grid_size as input to create a comparision graph.
```
python experiments/demo_scalability.py
```

## 📚 9. References
    - [Qiskit Optimization](https://qiskit.org/documentation/optimization/)
    - [D-Wave Ocean SDK](https://docs.ocean.dwavesys.com/)
    - Farhi et al., A Quantum Approximate Optimization Algorithm (2014)