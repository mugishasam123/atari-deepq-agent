# Atari Deep Q-Learning Agent

Formative 3 assignment: train and evaluate a DQN agent on an Atari game using **Stable Baselines3** and **Gymnasium**.

- **Environment:** `ALE/SpaceInvaders-v5`
- **Deliverables:** `train.py`, `play.py`, hyperparameter table, and gameplay video.

---

## Project structure

```
atari-deepq-agent/
├── train.py           # Train DQN; supports CnnPolicy/MlpPolicy and full hyperparameter CLI
├── play.py            # Load trained model and play with greedy policy + render
├── requirements.txt   # Python dependencies
├── models/            # Saved models (created on first run; add to .gitignore or keep best only)
├── logs/              # TensorBoard logs (created on first run)
└── README.md
```

---

## Setup

1. **Clone and enter the repo**
   ```bash
   cd atari-deepq-agent
   ```

2. **Create and activate a virtual environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate   # Windows: .\venv\Scripts\Activate.ps1
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Accept Atari ROM license** (if prompted when first using the env)
   - See [Gymnasium Atari](https://gymnasium.farama.org/environments/atari/) for ROM setup.

---

## Usage

### Training

Train with default or custom hyperparameters. Model and logs are saved under `models/` and `logs/` with names derived from the run config.

```bash
python train.py --steps 500000 [options]
```

**Options:** `--policy` (CnnPolicy|MlpPolicy), `--lr`, `--gamma`, `--batch_size`, `--epsilon_start`, `--epsilon_end`, `--epsilon_decay`, `--n_envs`. Use `--save_as_best` to also save the final model as `models/dqn_model.zip` (for submission).

**Example (recommended baseline):**
```bash
python train.py --steps 500000 --lr 2.5e-4 --gamma 0.99 --batch_size 32 --epsilon_decay 150000
```

**Compare policies:**
```bash
python train.py --policy CnnPolicy --steps 250000
python train.py --policy MlpPolicy --steps 250000
```

### Playing

Load a trained model and run episodes with greedy action selection (deterministic, no exploration). Game window opens via `render_mode="human"`.

```bash
python play.py --model models/dqn_model.zip --n_episodes 5
```

---

## Hyperparameter experiments (10 per member)

Each member runs **10 different hyperparameter combinations**, records them in the table below, and notes behavior (reward trend, stability, episode length, etc.). Use the same **total steps** for all runs (e.g. 250k or 500k) for fair comparison.

### Role split

| Member | Focus area | Responsibility |
|--------|------------|----------------|
| **Didier Ganza** | Gamma & learning rate interaction | Vary `gamma` and `lr` (and batch where relevant); document impact on long-term reward and stability. |
| **Edith Nyanjiru Githinji** | Learning rate & batch size | Vary `lr` and `batch_size`; document impact on learning speed and sample efficiency. |
| **Makuochi Prince Okoye** | Epsilon decay & epsilon_end | Vary `epsilon_decay` and `epsilon_end`; document exploration–exploitation balance. |
| **Mugisha Samuel** | Fine-tuning: lr & epsilon decay | Grid search on lr and epsilon_decay; document which combination performs best and why. |

---

### Didier Ganza – Gamma & learning rate interaction

**Mission:** Find the optimal discount factor (gamma) and its interaction with lr and batch_size.

| # | Hyperparameter set | Noted behavior |
|---|--------------------|----------------|
| 1 | lr=, gamma=, batch=, epsilon_start=, epsilon_end=, epsilon_decay= | |
| 2 | lr=, gamma=, batch=, epsilon_start=, epsilon_end=, epsilon_decay= | |
| 3 | lr=, gamma=, batch=, epsilon_start=, epsilon_end=, epsilon_decay= | |
| 4 | lr=, gamma=, batch=, epsilon_start=, epsilon_end=, epsilon_decay= | |
| 5 | lr=, gamma=, batch=, epsilon_start=, epsilon_end=, epsilon_decay= | |
| 6 | lr=, gamma=, batch=, epsilon_start=, epsilon_end=, epsilon_decay= | |
| 7 | lr=, gamma=, batch=, epsilon_start=, epsilon_end=, epsilon_decay= | |
| 8 | lr=, gamma=, batch=, epsilon_start=, epsilon_end=, epsilon_decay= | |
| 9 | lr=, gamma=, batch=, epsilon_start=, epsilon_end=, epsilon_decay= | |
| 10 | lr=, gamma=, batch=, epsilon_start=, epsilon_end=, epsilon_decay= | |

---

### Edith Nyanjiru Githinji – Learning rate & batch size

**Mission:** Explore the impact of learning rate and batch size combinations.

| # | Hyperparameter set | Noted behavior |
|---|--------------------|----------------|
| 1 | lr=1e-4, gamma=0.99, batch=16, epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=150000 | Fastest training (539 FPS, 7:43). Lower final reward (245). Loss 0.193. Smaller batch enables more frequent updates but noisier gradients, resulting in suboptimal performance. |
| 2 | lr=1e-4, gamma=0.99, batch=32, epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=150000 | Balanced performance (279 reward, 496 FPS, 8:13). Loss 0.119 - lowest among lr=1e-4 runs. Larger batch provides more stable gradient estimates, improving reward by +34 over batch=16. |
| 3 | lr=1e-4, gamma=0.99, batch=64, epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=150000 | Slower training (443 FPS, 9:03). Lower reward (255) than batch=32. Loss 0.344 - higher than batch=32. Larger batch size beyond 32 shows diminishing returns; may be underfitting with fixed learning rate. |
| 4 | lr=, gamma=, batch=, epsilon_start=, epsilon_end=, epsilon_decay= | |
| 5 | lr=, gamma=, batch=, epsilon_start=, epsilon_end=, epsilon_decay= | |
| 6 | lr=, gamma=, batch=, epsilon_start=, epsilon_end=, epsilon_decay= | |
| 7 | lr=, gamma=, batch=, epsilon_start=, epsilon_end=, epsilon_decay= | |
| 8 | lr=, gamma=, batch=, epsilon_start=, epsilon_end=, epsilon_decay= | |
| 9 | lr=, gamma=, batch=, epsilon_start=, epsilon_end=, epsilon_decay= | |
| 10 | lr=, gamma=, batch=, epsilon_start=, epsilon_end=, epsilon_decay= | |

---

### Makuochi Prince Okoye – Epsilon decay & epsilon_end

**Mission:** Optimize exploration vs exploitation by testing epsilon_decay and epsilon_end.

| # | Hyperparameter set | Noted behavior |
|---|--------------------|----------------|
| 1 | lr=1e-4, gamma=0.99, batch=64, eps_start=1.0, eps_end=0.05, eps_decay=50k | Space Invaders Test Run (150k steps). Mean reward +241. Rapid learning and dodging observed. |
| 2 | lr=2.5e-4, gamma=0.99, batch=32, eps_start=1.0, eps_end=0.20, eps_decay=250k | High Final Epsilon (0.20). Mean reward +193. Worse than baseline. Forced exploration prevented exploiting learned skills. |
| 3 | lr=2.5e-4, gamma=0.99, batch=32, eps_start=1.0, eps_end=0.01, eps_decay=250k | Low Final Epsilon (0.01). Mean reward +184. Decay was too slow (250k). Epsilon only reached 0.40 at the end of the 150k run, causing excessive exploration. |
| 4 | lr=2.5e-4, gamma=0.99, batch=32, eps_start=1.0, eps_end=0.05, eps_decay=50k | Fast Epsilon Decay (50k). Mean reward +251. Best result. Agent completed exploration early, allowing 100k steps of pure strategy exploitation. |
| 5 | lr=2.5e-4, gamma=0.99, batch=32, eps_start=1.0, eps_end=0.05, eps_decay=500k | Slow Epsilon Decay (500k). Mean reward +167. Agent was still exploring 71.5% of the time at the end of training, resulting in a very poor score. |
| 6 | lr=, gamma=, batch=, epsilon_start=, epsilon_end=, epsilon_decay= | |
| 7 | lr=, gamma=, batch=, epsilon_start=, epsilon_end=, epsilon_decay= | |
| 8 | lr=, gamma=, batch=, epsilon_start=, epsilon_end=, epsilon_decay= | |
| 9 | lr=, gamma=, batch=, epsilon_start=, epsilon_end=, epsilon_decay= | |
| 10 | lr=, gamma=, batch=, epsilon_start=, epsilon_end=, epsilon_decay= | |

---

### Mugisha Samuel – Fine-tuning: learning rate & epsilon decay

**Mission:** Grid search on lr and epsilon_decay; document which combinations perform best and why.

**Setup (all runs):** `ALE/SpaceInvaders-v5`, `CnnPolicy`, 500,000 timesteps, 4 parallel envs. Fixed unless noted: `gamma=0.99`, `batch_size=32`, `epsilon_start=1.0`, `epsilon_end=0.05`. Metrics from TensorBoard `rollout/ep_rew_mean` (mean episode reward).

| # | Hyperparameter set | Noted behavior |
|---|--------------------|----------------|
| 1 | lr=0.0001, gamma=0.99, batch=32, epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=250000 | Peak ep_rew_mean ~290.5; final ~258.2. Stable but slower learning than higher lr. |
| 2 | lr=0.00025, gamma=0.99, batch=32, epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=50000 | Peak ~318.4; final ~295.1. Fast epsilon decay helped exploitation. |
| 3 | lr=0.00025, gamma=0.99, batch=32, epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=100000 | Peak ~316.1; final ~296.9. Strong; similar to exp 2. |
| 4 | lr=0.00025, gamma=0.99, batch=32, epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=150000 | Peak ~337.6; final ~304.0. **Second best** — good balance of lr and decay. |
| 5 | lr=0.00025, gamma=0.99, batch=32, epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=250000 | Peak ~299.5; final ~299.5. Slower decay than 2–4; plateaued below best peaks. |
| 6 | lr=0.0005, gamma=0.99, batch=32, epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=50000 | Peak **~344.6**; final **~332.5**. **Best run** — high lr + fast decay matched Space Invaders well. |
| 7 | lr=0.0005, gamma=0.99, batch=32, epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=150000 | Peak ~299.5; final ~277.1. Same peak as exp 5 tier; decay 150k weaker than 50k at this lr. |
| 8 | lr=0.0001, gamma=0.99, batch=32, epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=50000 | Peak ~278.5; final ~257.0. Low lr limited gains even with fast decay. |
| 9 | lr=0.0001, gamma=0.99, batch=32, epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=150000 | Peak ~302.9; final ~273.3. Mid-tier; better than exp 1 & 8 for same lr. |
| 10 | lr=0.0003, gamma=0.99, batch=32, epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=100000 | Peak ~290.1; final ~237.1. Reward dipped late; less stable finish than top runs. |

**Best model (for `play.py`):** `models/policy-CnnPolicy_lr-0.0005_gamma-0.99_batch-32_eps1.0-0.05_decay-50000.zip`

**Insight:** Highest mean rewards came from **lr=5×10⁻⁴** with **epsilon_decay=50,000** (exploration ends early; most steps exploit). **lr=2.5×10⁻⁴** with **epsilon_decay=150,000** was the next best. Very slow decay or low lr at 500k steps underperformed relative to fast-decay configs.


---

## TensorBoard

Inspect training curves (e.g. reward, episode length):

```bash
tensorboard --logdir logs/
```

Open the URL shown (e.g. http://localhost:6006).

---



## References

- [Stable Baselines3 — DQN](https://stable-baselines3.readthedocs.io/en/stable/modules/dqn.html)
- [Gymnasium — Atari](https://gymnasium.farama.org/environments/atari/)
