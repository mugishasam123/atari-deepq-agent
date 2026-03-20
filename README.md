# Atari Deep Q-Learning Agent

Formative 3 assignment: train and evaluate a DQN agent on an Atari game using **Stable Baselines3** and **Gymnasium**.

- **Environment:** `ALE/Pong-v5`
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
