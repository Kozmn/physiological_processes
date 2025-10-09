# Modeling Physiological Processes - Course Assignments

## About This Repository

This repository contains my solutions for assignments from the "Modeling Physiological Processes" university course. The projects focus on applying mathematical and computational models to simulate and analyze various biological systems, particularly within the cardiovascular system. Each lab explores a different physiological concept through Python programming, data analysis, and visualization.

---

## Labs & Assignments

### Lab 1: Blood Vessel Compliance Analysis
* **Description:** This assignment involves fitting linear, exponential, and logistic models to patient pressure-volume (P-V) data to analyze vascular compliance. It also includes an implementation of the Windkessel model to simulate pressure decay in arteries.
* **Key Concepts:** P-V relationships, vascular compliance, model fitting (`scipy.optimize.curve_fit`), Windkessel model.
* **Files:** `lab1/list1.py`, `lab1/list1.ipynb`, `lab1/patients.csv`

### Lab 2: Blood Flow Modeling in the Circulatory System
* **Description:** This project focuses on modeling blood flow through a simplified arterial tree (aorta -> branches -> arterioles) using the Hagen-Poiseuille equation. It involves calculating the total hydrodynamic resistance of the network and performing sensitivity analyses on parameters like vessel radius and blood viscosity (simulating conditions like anemia).
* **Key Concepts:** Hagen-Poiseuille law, hydrodynamic resistance (series & parallel), sensitivity analysis, cardiovascular fluid dynamics.
* **Files:** `lab2/list2.py`, `lab2/list2.ipynb`

---

## Technologies Used
* Python 3
* NumPy for numerical operations
* Pandas for data manipulation (Lab 1)
* Matplotlib for data visualization
* SciPy for model fitting (Lab 1)
* Jupyter Notebook for interactive analysis

---

## How to Run

Each lab folder (`lab1/`, `lab2/`) contains both a Python script (`.py`) and a Jupyter Notebook (`.ipynb`) with the complete analysis. The notebooks provide a more descriptive, step-by-step walkthrough of the process and include the final plots.
