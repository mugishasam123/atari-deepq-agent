import subprocess
import os

# member: Makuochi Prince Okoye
# focus: Epsilon decay & epsilon_end
experiments = [
    {"lr": 2.5e-4, "gamma": 0.99, "batch": 32, "eps_start": 1.0, "eps_end": 0.05, "eps_decay": 250_000}, # Baseline
    {"lr": 2.5e-4, "gamma": 0.99, "batch": 32, "eps_start": 1.0, "eps_end": 0.20, "eps_decay": 250_000}, # High End
    {"lr": 2.5e-4, "gamma": 0.99, "batch": 32, "eps_start": 1.0, "eps_end": 0.01, "eps_decay": 250_000}, # Low End
    {"lr": 2.5e-4, "gamma": 0.99, "batch": 32, "eps_start": 1.0, "eps_end": 0.05, "eps_decay": 50_000},  # Fast Decay
    {"lr": 2.5e-4, "gamma": 0.99, "batch": 32, "eps_start": 1.0, "eps_end": 0.05, "eps_decay": 500_000}, # Slow Decay
    {"lr": 2.5e-4, "gamma": 0.99, "batch": 32, "eps_start": 1.0, "eps_end": 0.05, "eps_decay": 100_000}, # Med-Fast Decay
    {"lr": 2.5e-4, "gamma": 0.99, "batch": 32, "eps_start": 0.5, "eps_end": 0.05, "eps_decay": 250_000}, # Low start
    {"lr": 2.5e-4, "gamma": 0.99, "batch": 32, "eps_start": 1.0, "eps_end": 0.10, "eps_decay": 150_000}, # Combo 1
    {"lr": 2.5e-4, "gamma": 0.99, "batch": 32, "eps_start": 1.0, "eps_end": 0.02, "eps_decay": 350_000}, # Combo 2
    {"lr": 2.5e-4, "gamma": 0.99, "batch": 64, "eps_start": 1.0, "eps_end": 0.05, "eps_decay": 250_000}, # Batch 64 (comparison)
]

STEPS = 150_000 # Increased to 150k for Space Invaders

for i, exp in enumerate(experiments):
    print(f"\n>>> Running Experiment {i+1}/10 for Makuochi Prince Okoye...")
    cmd = [
        "python3", "train.py",
        "--steps", str(STEPS),
        "--lr", str(exp["lr"]),
        "--gamma", str(exp["gamma"]),
        "--batch_size", str(exp["batch"]),
        "--epsilon_start", str(exp["eps_start"]),
        "--epsilon_end", str(exp["eps_end"]),
        "--epsilon_decay", str(exp["eps_decay"]),
        "--buffer_size", "100000",
        "--learning_starts", "10000",
        "--n_envs", "4",
        "--device", "mps"
    ]
    if i == 0: # Save the first one (or last best) as dqn_model.zip
        cmd.append("--save_as_best")
        
    subprocess.run(cmd)

print("\nAll experiments for Makuochi Prince Okoye completed.")
