import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# Part 1: The Automated Summary
# ==========================================

def summarize_clinic(data):
    """
    Returns the Mean and Standard Deviation of the provided patient recovery data.
    """
    mean_val = np.mean(data)
    std_val = np.std(data)
    return mean_val, std_val

def run_part1():
    print("--- Part 1: Automated Summary ---")
    data = [4.2, 5.1, 3.8, 7.2, 5.5, 6.0]
    
    # 1. Summarize initial data
    mean, std = summarize_clinic(data)
    print(f"Initial Data: {data}")
    print(f"Mean: {mean:.2f}")
    print(f"Standard Deviation: {std:.2f}")
    
    # 2. Add outlier and re-run
    print("\nAdding outlier (15.0)...")
    data.append(15.0)
    mean_out, std_out = summarize_clinic(data)
    print(f"Updated Data: {data}")
    print(f"New Mean: {mean_out:.2f}")
    print(f"New Standard Deviation: {std_out:.2f}")
    print(f"Change in Std Dev: {std_out - std:.2f}")
    print("-" * 33 + "\n")

# ==========================================
# Part 2: Visualizing the "Compass"
# ==========================================

def run_part2():
    print("--- Part 2: Visualizing the 'Compass' ---")
    
    # Lab data from "Math for ML (1).ipynb"
    dosage = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    # Base recovery scores before noise (representing the general relationship)
    # recovery_base = np.array([5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10, 5.5, 6.2, 6.8, 7.3, 7.8, 8.3, 8.8, 9.3, 9.8, 10])
    
    # Simulated lab data with noise as seen in the notebook
    np.random.seed(42)
    recovery = np.array([5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10, 5.5, 6.2, 6.8, 7.3, 7.8, 8.3, 8.8, 9.3, 9.8, 10])
    recovery = recovery + np.random.normal(0, 0.5, size=len(dosage))
    recovery = np.clip(recovery, 1, 10)

    # 1. Create scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(dosage, recovery, color='blue', label='Lab Data (Actual)', alpha=0.7)
    
    # 2. Prediction line for w = 0.1
    # Line equation: y = w * x (assuming simple proportionality for this exercise)
    # Note: If intercept is assumed 0, a weight of 0.1 is very low. 
    # Let's check the context. The notebook shows recovery starting at ~5 for dosage 1.
    # Linear model: recovery = weight * dosage + bias.
    # But the prompt says "Prediction Line... weight of w=0.1". 
    # Usually in these simple examples, w might be a slope.
    
    x_range = np.linspace(0, 11, 100)
    
    # 3. Developer Challenge: Plot lines for w=0.2, 0.5, 0.8
    weights = [0.1, 0.2, 0.5, 0.8]
    colors = ['gray', 'green', 'orange', 'red']
    
    for w, c in zip(weights, colors):
        y_pred = w * x_range
        label = f'Prediction Line (w={w})'
        if w == 0.1:
            plt.plot(x_range, y_pred, color=c, linestyle='--', label=label)
        else:
            plt.plot(x_range, y_pred, color=c, label=label)

    plt.title('Clinical Dashboard: Dosage vs Recovery')
    plt.xlabel('Dosage (Units)')
    plt.ylabel('Recovery Score')
    plt.xlim(0, 11)
    plt.ylim(0, 11)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    
    # Save the plot
    plot_filename = 'dosage_vs_recovery.png'
    plt.savefig(plot_filename)
    print(f"Plot saved as {plot_filename}")
    plt.show() # This might not work in non-interactive environment, but savefig works.

    print("\nQuestion: Which weight 'fits' the dots best?")
    print("Answer: Visually, the line with w = 0.8 or closer to the data trend fits better,")
    print("although none of these pass through the origin-offset data perfectly (bias is needed).")
    print("However, among the choices [0.2, 0.5, 0.8], w=0.8 is the closest to the general trend of higher recovery.")
    print("-" * 38 + "\n")

if __name__ == "__main__":
    run_part1()
    run_part2()
