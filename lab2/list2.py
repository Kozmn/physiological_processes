import numpy as np
import matplotlib.pyplot as plt


ETA_SCENARIOS = {
    'Normal': 3.5e-3,
    'Anemia': 2.8e-3
}
MMHG_TO_PA = 133.322
PIN = 100 * MMHG_TO_PA
POUT = 10 * MMHG_TO_PA
NUM_TERMINALS = 4


class Vessel():
    """Represents a single blood vessel."""
    def __init__(self, L, r, eta):
        self.L = L
        self.r = r
        self.eta = eta
    
    @property
    def R(self):
        """Calculates the hydrodynamic resistance of the vessel."""
        return (8 * self.eta * self.L) / (np.pi * self.r**4)


def calculate_total_resistance(aorta, branches, terminals):
    """Calculates the total equivalent resistance for the entire vascular network."""
    R_term_eq = terminals[0].R / 2
    R_path1 = R_term_eq + branches[0].R
    R_parallel_paths = R_path1 / 2
    R_total = aorta.R + R_parallel_paths
    return R_total

def run_viscosity_analysis_and_plot(scenarios):
    """Runs the viscosity analysis for different scenarios and generates a bar chart."""
    print("\n--- Viscosity Analysis ---")
    
    results = {}
    for case_name, eta_value in scenarios.items():

        aorta = Vessel(L=0.3, r=0.012, eta=eta_value)
        branches = [Vessel(L=0.2, r=0.006, eta=eta_value) for _ in range(2)]
        terminals = [Vessel(L=0.02, r=0.0015, eta=eta_value) for _ in range(NUM_TERMINALS)]
        

        R_total = calculate_total_resistance(aorta, branches, terminals)
        Q_total = (PIN - POUT) / R_total
        Q_single_terminal = Q_total / NUM_TERMINALS
        
        results[case_name] = Q_single_terminal

    print(f"Perfusion results (m^3/s): {results}")


    names = list(results.keys())
    values_ml_s = [q * 1e6 for q in results.values()]

    plt.bar(names, values_ml_s, color=['skyblue', 'salmon'])
    plt.ylabel("Flow in arteriole (ml/s)")
    plt.title("Perfusion Comparison: Normal vs. Anemia")
    plt.show()

def run_sensitivity_analysis(base_aorta, base_branches, base_terminals):
    """Runs the sensitivity analysis for changes in terminal radius."""
    print("\n--- Radius Sensitivity Analysis ---")
    
    # Baseline case
    R_base = calculate_total_resistance(base_aorta, base_branches, base_terminals)
    Q_base = (PIN - POUT) / R_base
    
    # -10% radius scenario
    r_smaller = base_terminals[0].r * 0.9
    terminals_smaller = [Vessel(L=0.02, r=r_smaller, eta=base_terminals[0].eta) for _ in range(NUM_TERMINALS)]
    R_smaller = calculate_total_resistance(base_aorta, base_branches, terminals_smaller)
    Q_smaller = (PIN - POUT) / R_smaller

    # +10% radius scenario
    r_bigger = base_terminals[0].r * 1.1
    terminals_bigger = [Vessel(L=0.02, r=r_bigger, eta=base_terminals[0].eta) for _ in range(NUM_TERMINALS)]
    R_bigger = calculate_total_resistance(base_aorta, base_branches, terminals_bigger)
    Q_bigger = (PIN - POUT) / R_bigger

    print(f"Baseline flow: {Q_base * 1e6:.2f} ml/s")
    print(f"Flow (-10% r): {Q_smaller * 1e6:.2f} ml/s")
    print(f"Flow (+10% r): {Q_bigger * 1e6:.2f} ml/s")


if __name__ == "__main__":
    # Create a baseline set of vessels for the 'Normal' viscosity case
    base_eta = ETA_SCENARIOS['Normal']
    aorta_base = Vessel(L=0.3, r=0.012, eta=base_eta)
    branches_base = [Vessel(L=0.2, r=0.006, eta=base_eta) for _ in range(2)]
    terminals_base = [Vessel(L=0.02, r=0.0015, eta=base_eta) for _ in range(NUM_TERMINALS)]
    

    run_sensitivity_analysis(aorta_base, branches_base, terminals_base)
    run_viscosity_analysis_and_plot(ETA_SCENARIOS)