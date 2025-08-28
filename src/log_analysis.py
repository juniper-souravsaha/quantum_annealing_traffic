import pandas as pd
import matplotlib.pyplot as plt

def analyze_log(log_csv="sa_log.csv", out_png="sa_plot.png"):
    # Load the CSV
    df = pd.read_csv(log_csv)

    # Ensure numeric types
    for col in ["episode", "temp", "current_cost", "best_cost", "acceptance_rate", "violations"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    fig, ax1 = plt.subplots(figsize=(8, 5))

    # Plot costs
    ax1.plot(df["episode"], df["current_cost"], label="Current Cost", alpha=0.6, color="blue")
    ax1.plot(df["episode"], df["best_cost"], label="Best Cost", linewidth=2, color="navy")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Cost", color="blue")
    ax1.tick_params(axis="y", labelcolor="blue")

    # Plot violations on secondary axis
    ax2 = ax1.twinx()
    ax2.plot(df["episode"], df["violations"], label="Violations", color="red", linewidth=2, linestyle="--")
    ax2.set_ylabel("Violations", color="red")
    ax2.tick_params(axis="y", labelcolor="red")

    # Combine legends
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="upper right")

    plt.title("Simulated Annealing: Cost & Violations Over Time")
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()
    print(f"[log_analysis] Plot saved to {out_png}")


if __name__ == "__main__":
    analyze_log()