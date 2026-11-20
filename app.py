import time
import streamlit as st
import numpy as np
import pandas as pd
from cliff_walking import train_comparison

st.set_page_config(
    page_title="Q-Learning vs. SARSA - Cliff Walking",
    page_icon="🧗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .cliff-tile {
        background-color: #450a0a;
        color: #f87171;
        font-weight: bold;
        text-align: center;
        border-radius: 4px;
        padding: 6px 1px;
        font-size: 0.8rem;
        margin: 1px;
        border: 1px solid #7f1d1d;
    }
    .safe-tile {
        background-color: #0f172a;
        color: #94a3b8;
        font-weight: bold;
        text-align: center;
        border-radius: 4px;
        padding: 6px 1px;
        font-size: 0.8rem;
        margin: 1px;
        border: 1px solid #1e293b;
    }
    .agent-q-tile {
        background-color: #ef4444;
        color: #ffffff;
        font-weight: bold;
        text-align: center;
        border-radius: 4px;
        padding: 6px 1px;
        font-size: 0.85rem;
        margin: 1px;
    }
    .agent-sarsa-tile {
        background-color: #0ea5e9;
        color: #ffffff;
        font-weight: bold;
        text-align: center;
        border-radius: 4px;
        padding: 6px 1px;
        font-size: 0.85rem;
        margin: 1px;
    }
</style>
""", unsafe_allow_html=True)

# Session state initialization
if "results" not in st.session_state:
    st.session_state.results = None
if "step_q" not in st.session_state:
    st.session_state.step_q = 0
if "step_sarsa" not in st.session_state:
    st.session_state.step_sarsa = 0

# ----------------- SIDEBAR -----------------
st.sidebar.title("🧗 Cliff Walking")
st.sidebar.markdown(
    "**Battle of Algorithms:**\n"
    "• **Q-Learning:** Off-Policy (The Daredevil)\n"
    "• **SARSA:** On-Policy (The Cautious Explorer)"
)

st.sidebar.subheader("Hyperparameters")
episodes = st.sidebar.slider("Training Episodes", min_value=100, max_value=1000, value=500, step=50)
alpha = st.sidebar.slider("Learning Rate (α)", min_value=0.1, max_value=1.0, value=0.5, step=0.05)
gamma = st.sidebar.slider("Discount Factor (γ)", min_value=0.8, max_value=1.0, value=1.0, step=0.05)
epsilon = st.sidebar.slider("Exploration Rate (ε)", min_value=0.01, max_value=0.30, value=0.10, step=0.01)

if st.sidebar.button("🚀 Train Both Agents", type="primary", use_container_width=True):
    with st.spinner("Simulating Q-Learning vs SARSA on Cliff Walking..."):
        res = train_comparison(episodes=episodes, alpha=alpha, gamma=gamma, epsilon=epsilon)
        st.session_state.results = res
        st.session_state.step_q = 0
        st.session_state.step_sarsa = 0
        st.sidebar.success("Training complete!")

if st.sidebar.button("🔄 Reset", use_container_width=True):
    st.session_state.results = None
    st.session_state.step_q = 0
    st.session_state.step_sarsa = 0
    st.rerun()

# Helper grid renderer
def render_board(agent_pos, path, is_q_learning=True):
    for r in range(4):
        cols = st.columns(12)
        for c in range(12):
            state = r * 12 + c
            is_agent = (state == agent_pos)
            is_start = (state == 36)
            is_goal = (state == 47)
            is_cliff = (37 <= state <= 46)
            is_visited = state in path

            if is_agent:
                cls = "agent-q-tile" if is_q_learning else "agent-sarsa-tile"
                lbl = "🤖"
            elif is_goal:
                cls = "safe-tile"
                lbl = "🏆 G"
            elif is_start:
                cls = "safe-tile"
                lbl = "🟦 S"
            elif is_cliff:
                cls = "cliff-tile"
                lbl = "🕳️"
            elif is_visited:
                cls = "safe-tile"
                lbl = "•"
            else:
                cls = "safe-tile"
                lbl = "."

            cols[c].markdown(f'<div class="{cls}">{lbl}</div>', unsafe_allow_html=True)

# ----------------- MAIN VIEW -----------------
st.title("🧗 Q-Learning vs. SARSA on Cliff Walking")
st.markdown(
    "The definitive Reinforcement Learning showdown: Does the agent take the optimal, dangerous path right on the cliff edge "
    "(**Q-Learning**), or does it take the cautious detour around the top (**SARSA**)? Explore live below."
)

tab1, tab2, tab3, tab4 = st.tabs([
    "⚔️ Side-by-Side Path Showdown",
    "📊 Sutton & Barto Reward Curves",
    "🧭 Policy Direction Maps",
    "📖 The Deep Theory (On vs Off Policy)"
])

# ---------- TAB 1: SIDE-BY-SIDE PATHS ----------
with tab1:
    if st.session_state.results is None:
        st.info("👈 Click **'🚀 Train Both Agents'** in the sidebar to simulate both algorithms.")
    else:
        res = st.session_state.results
        
        # High level metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Q-Learning Path Length", f"{len(res['q_path'])-1} steps", delta="Optimal Shortest", delta_color="normal")
        m2.metric("SARSA Path Length", f"{len(res['sarsa_path'])-1} steps", delta="Safe Detour", delta_color="normal")
        m3.metric("Q-Learning Cliff Falls", f"{res['q_falls']} falls", delta="High Risk", delta_color="inverse")
        m4.metric("SARSA Cliff Falls", f"{res['sarsa_falls']} falls", delta="Low Risk", delta_color="normal")

        st.markdown("---")
        
        # Simulation step controls
        c_ctrl1, c_ctrl2, c_ctrl3 = st.columns([1, 1, 2])
        with c_ctrl1:
            if st.button("▶️ Step Both Agents", use_container_width=True):
                if st.session_state.step_q < len(res['q_path']) - 1:
                    st.session_state.step_q += 1
                if st.session_state.step_sarsa < len(res['sarsa_path']) - 1:
                    st.session_state.step_sarsa += 1
                st.rerun()
        with c_ctrl2:
            if st.button("🔄 Reset Animation", use_container_width=True):
                st.session_state.step_q = 0
                st.session_state.step_sarsa = 0
                st.rerun()
                
        col_q, col_sarsa = st.columns(2)
        
        with col_q:
            st.markdown("### 🔴 Q-Learning (Off-Policy)")
            st.markdown("*Walks right on the edge (Row 2). Shortest path, but high risk of accidental falls during exploration.*")
            q_cur = res['q_path'][st.session_state.step_q]
            render_board(q_cur, res['q_path'][:st.session_state.step_q], is_q_learning=True)
            st.caption(f"Step {st.session_state.step_q} of {len(res['q_path'])-1} | Current State: {q_cur}")

        with col_sarsa:
            st.markdown("### 🔵 SARSA (On-Policy)")
            st.markdown("*Goes all the way up to Row 0. Takes the long way, but avoids fatal -100 cliff penalties.*")
            sarsa_cur = res['sarsa_path'][st.session_state.step_sarsa]
            render_board(sarsa_cur, res['sarsa_path'][:st.session_state.step_sarsa], is_q_learning=False)
            st.caption(f"Step {st.session_state.step_sarsa} of {len(res['sarsa_path'])-1} | Current State: {sarsa_cur}")

# ---------- TAB 2: REWARD CURVES ----------
with tab2:
    st.subheader("Training Reward Comparison (Sutton & Barto Fig 6.4)")
    if st.session_state.results is None:
        st.info("Train the agents to view the reward curves.")
    else:
        res = st.session_state.results
        
        # 10-episode rolling average
        window = 15
        q_smooth = pd.Series(res['q_rewards']).rolling(window, min_periods=1).mean()
        sarsa_smooth = pd.Series(res['sarsa_rewards']).rolling(window, min_periods=1).mean()

        df_rewards = pd.DataFrame({
            "Q-Learning (Off-Policy)": q_smooth,
            "SARSA (On-Policy)": sarsa_smooth
        })
        
        st.line_chart(df_rewards)
        st.caption("Notice how SARSA achieves a significantly **higher online return during training** (around -20 to -30), while Q-Learning frequently plummets to -100 because its exploratory moves drop it off the cliff edge.")

# ---------- TAB 3: POLICY DIRECTION MAPS ----------
with tab3:
    st.subheader("Learned Greedy Policy Arrows")
    if st.session_state.results is None:
        st.info("Train the agents to inspect their policy maps.")
    else:
        res = st.session_state.results
        action_arrows = {0: "↑", 1: "→", 2: "↓", 3: "←"}

        col_p1, col_p2 = st.columns(2)
        with col_p1:
            st.markdown("#### Q-Learning Policy")
            for r in range(4):
                cols = st.columns(12)
                for c in range(12):
                    st_idx = r * 12 + c
                    if 37 <= st_idx <= 46:
                        txt = "🕳️"
                    elif st_idx == 47:
                        txt = "🏆"
                    else:
                        best_a = int(np.argmax(res['q_agent'].Q[st_idx, :]))
                        txt = action_arrows[best_a]
                    cols[c].button(txt, key=f"q_pol_{st_idx}", use_container_width=True)

        with col_p2:
            st.markdown("#### SARSA Policy")
            for r in range(4):
                cols = st.columns(12)
                for c in range(12):
                    st_idx = r * 12 + c
                    if 37 <= st_idx <= 46:
                        txt = "🕳️"
                    elif st_idx == 47:
                        txt = "🏆"
                    else:
                        best_a = int(np.argmax(res['sarsa_agent'].Q[st_idx, :]))
                        txt = action_arrows[best_a]
                    cols[c].button(txt, key=f"s_pol_{st_idx}", use_container_width=True)

# ---------- TAB 4: DEEP THEORY ----------
with tab4:
    st.subheader("Why Do They Choose Different Paths?")
    st.markdown("""
    ### 1. The Core Update Equations
    * **Q-Learning (Off-Policy):**
    """)
    st.latex(r"Q(S, A) \leftarrow Q(S, A) + \alpha \Big[ R + \gamma \max_{a} Q(S', a) - Q(S, A) \Big]")
    st.markdown("""
    * **SARSA (On-Policy):**
    """)
    st.latex(r"Q(S, A) \leftarrow Q(S, A) + \alpha \Big[ R + \gamma Q(S', A') - Q(S, A) \Big]")
    st.markdown("""
    ### 2. The Behavior Explained
    * **Q-Learning learns the optimal policy directly:** Because of the $\\max_a$ operator, Q-Learning ignores the fact that its $\\epsilon$-greedy exploration might choose an accidental step into the cliff. It learns the **theoretical shortest path (13 steps)** right along the edge.
    * **SARSA learns the optimal policy FOR an exploratory agent:** Because it updates with the *actual next action* $A'$, it realizes that walking near the edge has an $\\epsilon$-probability of stepping off into the cliff ($-100$). Therefore, the value of edge states drops, and SARSA naturally chooses the **cautious detour around the top (17 steps)**!
    """)
