import gymnasium as gym
import numpy as np

class QLearningCliffAgent:
    """
    Off-Policy Q-Learning Agent for Cliff Walking.
    Assumes greedy future actions (max_a).
    """
    def __init__(self, n_states=48, n_actions=4, alpha=0.5, gamma=1.0, epsilon=0.1):
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.Q = np.zeros((n_states, n_actions), dtype=np.float64)

    def get_action(self, state, greedy=False):
        if not greedy and np.random.uniform(0, 1) < self.epsilon:
            return np.random.randint(self.n_actions)
        max_q = np.max(self.Q[state, :])
        best_actions = np.where(self.Q[state, :] == max_q)[0]
        return int(np.random.choice(best_actions))

    def update(self, state, action, reward, next_state):
        best_next_q = np.max(self.Q[next_state, :])
        td_target = reward + self.gamma * best_next_q
        self.Q[state, action] += self.alpha * (td_target - self.Q[state, action])

class SarsaCliffAgent:
    """
    On-Policy SARSA Agent for Cliff Walking.
    Accounts for actual exploratory next actions (Q[next_state, next_action]).
    """
    def __init__(self, n_states=48, n_actions=4, alpha=0.5, gamma=1.0, epsilon=0.1):
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.Q = np.zeros((n_states, n_actions), dtype=np.float64)

    def get_action(self, state, greedy=False):
        if not greedy and np.random.uniform(0, 1) < self.epsilon:
            return np.random.randint(self.n_actions)
        max_q = np.max(self.Q[state, :])
        best_actions = np.where(self.Q[state, :] == max_q)[0]
        return int(np.random.choice(best_actions))

    def update(self, state, action, reward, next_state, next_action):
        td_target = reward + self.gamma * self.Q[next_state, next_action]
        self.Q[state, action] += self.alpha * (td_target - self.Q[state, action])

def train_comparison(episodes=500, alpha=0.5, gamma=1.0, epsilon=0.1, max_steps=500):
    """
    Trains both Q-Learning and SARSA on identical seeds/episodes
    and returns rewards, cliff fall statistics, and optimal paths.
    """
    q_agent = QLearningCliffAgent(alpha=alpha, gamma=gamma, epsilon=epsilon)
    sarsa_agent = SarsaCliffAgent(alpha=alpha, gamma=gamma, epsilon=epsilon)

    env_q = gym.make("CliffWalking-v1")
    env_s = gym.make("CliffWalking-v1")

    q_rewards = []
    sarsa_rewards = []
    q_cliff_falls = 0
    sarsa_cliff_falls = 0

    for ep in range(episodes):
        # --- Train Q-Learning ---
        s_q, _ = env_q.reset()
        ep_rew_q = 0
        for _ in range(max_steps):
            a_q = q_agent.get_action(s_q, greedy=False)
            ns_q, r_q, term_q, trunc_q, _ = env_q.step(a_q)
            q_agent.update(s_q, a_q, r_q, ns_q)
            s_q = ns_q
            ep_rew_q += r_q
            if r_q == -100:
                q_cliff_falls += 1
            if term_q or trunc_q:
                break
        q_rewards.append(ep_rew_q)

        # --- Train SARSA ---
        s_s, _ = env_s.reset()
        a_s = sarsa_agent.get_action(s_s, greedy=False)
        ep_rew_s = 0
        for _ in range(max_steps):
            ns_s, r_s, term_s, trunc_s, _ = env_s.step(a_s)
            na_s = sarsa_agent.get_action(ns_s, greedy=False)
            sarsa_agent.update(s_s, a_s, r_s, ns_s, na_s)
            s_s = ns_s
            a_s = na_s
            ep_rew_s += r_s
            if r_s == -100:
                sarsa_cliff_falls += 1
            if term_s or trunc_s:
                break
        sarsa_rewards.append(ep_rew_s)

    env_q.close()
    env_s.close()

    # Get Greedy Paths
    q_path = get_greedy_path(q_agent)
    sarsa_path = get_greedy_path(sarsa_agent)

    return {
        "q_agent": q_agent,
        "sarsa_agent": sarsa_agent,
        "q_rewards": q_rewards,
        "sarsa_rewards": sarsa_rewards,
        "q_falls": q_cliff_falls,
        "sarsa_falls": sarsa_cliff_falls,
        "q_path": q_path,
        "sarsa_path": sarsa_path
    }

def get_greedy_path(agent, max_steps=50):
    """
    Extracts the path taken under the greedy policy (no exploration).
    """
    env = gym.make("CliffWalking-v1")
    state, _ = env.reset()
    path = [state]
    
    for _ in range(max_steps):
        action = agent.get_action(state, greedy=True)
        next_state, reward, term, trunc, _ = env.step(action)
        path.append(next_state)
        state = next_state
        if term or trunc:
            break
    env.close()
    return path
