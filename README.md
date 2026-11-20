# 🧗 Q-Learning vs. SARSA on Cliff Walking

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-v0.29%2B-brightgreen.svg)](https://gymnasium.farama.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A head-to-head comparison between **Off-Policy Q-Learning** and **On-Policy SARSA** on Farama Gymnasium's **CliffWalking-v1** environment. Features an interactive **Streamlit** dashboard to visually demonstrate the difference between the optimal risky path and the safe detour.

---

## 📌 Problem Overview

Cliff Walking is a $4 \times 12$ gridworld:

```text
Row 0:  .    .    .    .    .    .    .    .    .    .    .    .
Row 1:  .    .    .    .    .    .    .    .    .    .    .    .
Row 2:  .    .    .    .    .    .    .    .    .    .    .    .
Row 3: [S] [THE CLIFF: -100 Penalty and Respawn at S]    [G]
```

* **Start:** `(3, 0)` [State 36]
* **Goal:** `(3, 11)` [State 47]
* **The Cliff:** States 37 to 46. Falling in gives $-100$ reward and resets to Start.
* **Standard step:** $-1$ reward.

---

## ⚔️ The Showdown: Q-Learning vs. SARSA

| Metric | 🔴 Q-Learning (Off-Policy) | 🔵 SARSA (On-Policy) |
| :--- | :---: | :---: |
| **Philosophy** | The Daredevil | The Cautious Strategist |
| **Path Taken** | **13 steps** (along the cliff edge on Row 2) | **17 steps** (safe detour along the top Row 0) |
| **Update Formula** | $R + \gamma \max_a Q(S', a)$ | $R + \gamma Q(S', A')$ |
| **Cliff Falls During Training** | **High (~150+ falls)** | **Low (~50–60 falls)** |
| **Online Learning Reward** | Lower (plummets to $-100$ frequently) | Higher (smoothly stays around $-20$ to $-30$) |

---

## 🚀 Interactive Streamlit Dashboard Features

* **Side-by-Side Path Showdown:** Watch both agents step through their routes simultaneously on synchronized $4 \times 12$ grid boards.
* **Sutton & Barto Fig 6.4 Reward Curves:** Live smoothed reward charts showing online return differences.
* **Policy Direction Heatmaps:** Arrow indicators showing the exact greedy policy learned by each agent.
* **Full Theory Tab:** In-depth explanation of the $\max_a$ operator vs. on-policy action evaluation.

---

## 💻 Quickstart & Running

### 1. Clone & Install
```bash
git clone https://github.com/Syed-Muhammad-Qamran/cliff-walking-comparison.git
cd cliff-walking-comparison
pip install -r requirements.txt
```

### 2. Launch the Streamlit App
```bash
streamlit run app.py
```

---

## 👤 Author
* **Syed Muhammad Qamran**
* GitHub: [@Syed-Muhammad-Qamran](https://github.com/Syed-Muhammad-Qamran)
