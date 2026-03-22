# Atari Deep Q-Learning Agent

Formative 3 assignment: train and evaluate a DQN agent on an Atari game using **Stable Baselines3** and **Gymnasium**.

- **Environment:** `ALE/SpaceInvaders-v5`
- **Deliverables:** `train.py`, `play.py`, hyperparameter table, and gameplay video.
- **Gameplay demo (YouTube):** [Trained agent playing Space Invaders (`play.py`)](https://youtu.be/15m1110atqg)

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

**Options:** `--policy` (CnnPolicy|MlpPolicy), `--lr`, `--gamma`, `--batch_size`, `--epsilon_start`, `--epsilon_end`, `--epsilon_decay`, `--n_envs`, `--buffer_size`. Use `--save_as_best` to also save the final model as `models/dqn_model.zip` (for submission).

**Example (recommended baseline):**
```bash
python train.py --steps 500000 --lr 2.5e-4 --gamma 0.99 --batch_size 32 --epsilon_decay 50000 --save_as_best
```

**Compare policies:**
```bash
python train.py --policy CnnPolicy --steps 500000
python train.py --policy MlpPolicy --steps 500000
```

### Playing

Load a trained model and run episodes with greedy action selection (deterministic, no exploration). Game window opens via `render_mode="human"`.

```bash
python play.py --model models/dqn_model.zip --n_episodes 5
```

**Demo video:** [Screen recording of `play.py` with the trained DQN](https://youtu.be/15m1110atqg)

---

## Hyperparameter experiments (10 per member)

Each member runs **10 different hyperparameter combinations**, records them in the table below, and notes behavior (reward trend, stability, episode length, etc.). All runs use **500,000 total steps** for fair comparison.

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

| # | Hyperparameter set                            | Noted behavior |
|---|-----------------------------------------------|----------------|
| 1 |lr=0.00025, gamma=0.95, batch=64, epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=150000 (250k steps) | ep_rew_mean=282, best=303. Reduced gamma (0.95) discounts future rewards more heavily. At 250k steps the agent is still learning — good mid-training baseline showing gamma < 0.99 converges faster early on.               |
| 2 | lr=0.00025, gamma=0.95, batch=64, epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=150000 (500k steps) | ep_rew_mean=446, best=473. Best reward overall. Doubling steps with gamma=0.95 gave the largest jump (+164 mean). Shows that moderate gamma pairs well with this lr when given enough training time.               |
| 3 | lr=0.001, gamma=0.9, batch=128, epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=150000 | ep_rew_mean=447, reward=470. Low gamma (0.9) + high lr + large batch. Strong performance despite heavy discounting — large batch stabilised the aggressive learning rate. Fastest wall-clock time among the longer runs (1:51).               |
| 4 |lr=0.0005, gamma=0.8, batch=32, epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=150000 | ep_rew_mean=381, best=400. Very low gamma (0.8) — agent is near-sighted and only values immediate rewards. Decent result suggests the game's short-horizon structure (invader lanes) partially suits greedy planning, but the score ceiling is clearly limited.               |
| 5 |lr=0.005, gamma=0.99, batch=32, epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=150000| ep_rew_mean=210, reward=254. Very high lr (0.005) caused unstable updates despite a standard gamma. Loss (0.151) and low reward confirm the learning rate overwhelmed the gradient signal — gamma alone could not compensate for overshooting.               |
| 6 | lr=0.0001, gamma=0.99, batch=64, epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=150000 | ep_rew_mean=290, best=290. Standard gamma with conservative lr. Stable but slow convergence — the agent learned reliably without diverging. Low lr capped the reward ceiling; gamma=0.99 needed a higher lr to leverage long-horizon planning effectively.               |
| 7 | lr=0.0005, gamma=0.99, batch=128, epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=150000 | ep_rew_mean=408, best=434. Good synergy between high gamma and moderate lr. Large batch (128) smoothed gradients enough to handle gamma=0.99's sensitivity to Q-value overestimation. Strong second-tier performer.               |
| 8 | lr=0.01, gamma=0.99, batch=16, epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=150000 |ep_rew_mean=219. Extreme lr (0.01) with small batch and high gamma — worst combination. High gamma amplifies overestimation errors, and the very high lr compounded instability. Small batch gave noisy gradients; fastest FPS (196 it/s) but worst quality learning.                |
| 9 | lr=0.0005, gamma=0.99, batch=16, epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=150000 |ep_rew_mean=258, reward=256. Same lr and gamma as run 7 but batch=16 instead of 128. The drop from 408 → 258 highlights how critical batch size is when gamma=0.99: noisy gradients from a tiny batch destabilise value estimates and suppress performance.                |
| 10 |lr=0.0001, gamma=0.9, batch=64, epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=150000|ep_rew_mean=364, best=390. Low gamma + low lr: more conservative in both discounting and update magnitude. Mid-tier result — reduced gamma helped avoid overestimation at the cost of long-term credit assignment; pairing with a slightly higher lr would likely push this further.                |

---

### Edith Nyanjiru Githinji – Learning rate & batch size

**Mission:** Explore the impact of learning rate and batch size combinations.

| # | Hyperparameter set | Noted behavior |
|---|--------------------|----------------|
| 1 | lr=1e-4, gamma=0.99, batch=16, epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=150000 | Fastest training (539 FPS, 7:43). Lower final reward (245). Loss 0.193. Smaller batch enables more frequent updates but noisier gradients, resulting in suboptimal performance. |
| 2 | lr=1e-4, gamma=0.99, batch=32, epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=150000 | Balanced performance (279 reward, 496 FPS, 8:13). Loss 0.119 - lowest among lr=1e-4 runs. Larger batch provides more stable gradient estimates, improving reward by +34 over batch=16. |
| 3 | lr=1e-4, gamma=0.99, batch=64, epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=150000 | Slower training (443 FPS, 9:03). Lower reward (255) than batch=32. Loss 0.344 - higher than batch=32. Larger batch size beyond 32 shows diminishing returns; may be underfitting with fixed learning rate. |
| 4 | lr=2.5e-4, gamma=0.99, batch=16, epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=150000 | Fast training (535 FPS, 7:47). Surprisingly low reward (223) - worst among all runs so far. Extremely low loss (0.0511) indicates possible underfitting. Higher learning rate with small batch may cause unstable updates. |
| 5 | lr=2.5e-4, gamma=0.99, batch=32, epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=150000 | Baseline. High reward (290). Slowest training (58 FPS, 17:40). Higher loss (0.971). Learning rate 2.5x larger enables faster convergence but at cost of slower wall-clock time. |
| 6 | lr=2.5e-4, gamma=0.99, batch=64, epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=150000 | Moderate training speed (447 FPS, 9:18). Low reward (226) - second worst. Loss 0.309. Similar poor performance to batch=16, suggesting batch=32 is the optimal size for this learning rate. |
| 7 | lr=5e-4, gamma=0.99, batch=16, epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=150000 | Fast training (521 FPS, 7:59). Very low reward (204). Extremely low loss (0.0676). Higher learning rate (5e-4) with batch=16 completely fails to learn effectively. Model likely stuck in poor local optimum. |
| 8 | lr=5e-4, gamma=0.99, batch=32, epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=150000 | Strong performer (292 reward). Moderate training speed (260-489 FPS, 8:15). Very low loss (0.0577). Learning rate 5e-4 with batch=32 achieves excellent balance. |
| 9 | lr=5e-4, gamma=0.99, batch=64, epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=150000 |  BEST PERFORMER! Outstanding reward (338) - significantly higher than all others! Moderate training speed (452 FPS, 9:12). Loss 0.41. Larger batch size (64) with learning rate 5e-4 enables much better learning, possibly due to more stable gradient estimates. |
| 10 | lr=1e-3, gamma=0.99, batch=32, epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=150000 | Performance collapse! Very low reward (207). High loss (0.856). Learning rate too high (1e-3) causes unstable training; model fails to converge. Demonstrates importance of appropriate learning rate selection. |

---

### Makuochi Prince Okoye – Epsilon decay & epsilon_end

**Mission:** Optimize exploration vs exploitation by testing epsilon_decay, epsilon_end, buffer size, and learning rate on Space Invaders (500k steps each).

| # | Hyperparameter set | Mean Reward | Noted behavior |
|---|-------------------|-------------|----------------|
| 1 | lr=2.5e-4, batch=32, eps_end=0.05, eps_decay=250k | +288 | Baseline. Solid performance with balanced exploration/exploitation schedule. |
| 2 | lr=2.5e-4, batch=32, eps_end=0.20, eps_decay=250k | +250 | High Final Epsilon. Forced 20% randomness capped performance below baseline. |
| 3 | lr=1e-4, batch=64, eps_end=0.01, eps_decay=50k | +155 | Low LR + Large Batch. Loss exploded to 2240 — catastrophic divergence. |
| 4 | lr=2.5e-4, batch=32, eps_end=0.05, eps_decay=50k | **+364** | **Best result!** Fast decay gave 450k steps of pure exploitation. |
| 5 | lr=5e-4, batch=32, eps_end=0.02, eps_decay=50k | +297 | Higher LR caused slight instability vs the 2.5e-4 sweet spot. |
| 6 | lr=2.5e-4, batch=32, eps_end=0.05, eps_decay=100k | +254 | Med-fast decay. Slightly worse than fast 50k decay variant. |
| 7 | lr=2.5e-4, batch=32, eps_end=0.05, eps_decay=50k, buffer=500k | +314 | Large buffer (500k). Deeper memory broke 300 for the first time. |
| 8 | lr=2.5e-4, batch=64, eps_end=0.05, eps_decay=50k, buffer=500k | +308 | Batch 64 + large buffer. Stable gradients but slightly slower than batch=32. |
| 9 | lr=2.5e-4, batch=32, eps_end=0.05, eps_decay=50k | +290 | EvalCallback run. Peak model saved via checkpointing during training. |
| 10 | lr=3e-4, batch=32, eps_end=0.02, eps_decay=30k, buffer=500k | +310 | Ultra-fast decay + all optimizations. Strong but slightly unstable at higher LR. |

**Key findings:**
- **Fast epsilon decay (50k)** consistently outperforms slow decay — more exploitation time = higher scores.
- **Buffer size matters:** increasing from 100k to 500k frames pushed scores from ~290 to 310+.
- **lr=2.5e-4** is the sweet spot — both 1e-4 (too slow) and 5e-4 (too fast) perform worse.
- **Best model: Experiment 4** (+364) using fast decay with the proven learning rate.

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

**Best model among Mugisha’s 10 runs (optional `play.py` checkpoint):** `models/policy-CnnPolicy_lr-0.0005_gamma-0.99_batch-32_eps1.0-0.05_decay-50000.zip`

**Video (gameplay with trained agent):** [YouTube — `play.py` / Space Invaders](https://youtu.be/15m1110atqg)

**Insight:** Highest mean rewards came from **lr=5×10⁻⁴** with **epsilon_decay=50,000** (exploration ends early; most steps exploit). **lr=2.5×10⁻⁴** with **epsilon_decay=150,000** was the next best. Very slow decay or low lr at 500k steps underperformed relative to fast-decay configs.

---

## Group best model (maximum reward)

Across **all team members’** documented experiments, the **highest reward** came from **Didier Ganza** (experiment **#2** in his table). We treat this checkpoint as the **group’s best model** because it reported the **maximum** training metrics in our README: **ep_rew_mean ≈ 446** and peak **≈ 473** (TensorBoard / rollout logs), above the other members’ reported means.

**Hyperparameters for that run**

| Setting | Value |
|--------|--------|
| Policy | `CnnPolicy` |
| Training steps | **500,000** |
| `lr` | **0.00025** |
| `gamma` | **0.95** |
| `batch_size` | **64** |
| `epsilon_start` | 1.0 |
| `epsilon_end` | 0.05 |
| `epsilon_decay` | 150000 |

**Saved weights:** `models/policy-CnnPolicy_lr-0.00025_gamma-0.95_batch-64_eps1.0-0.05_decay-150000.zip`

```bash
python play.py --model models/policy-CnnPolicy_lr-0.00025_gamma-0.95_batch-64_eps1.0-0.05_decay-150000.zip --n_episodes 5
```

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


