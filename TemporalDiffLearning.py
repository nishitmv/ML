import numpy as np
import matplotlib.pyplot as plt


# --- 1. ENVIRONMENT: Windy Gridworld (Based on Example 6.5) ---

class WindyGrid:
    def __init__(self, rows=7, cols=10, start=(3, 0), goal=(3, 7)):
        self.rows = rows
        self.cols = cols
        self.start_state = start
        self.goal_state = goal
        self.current_state = start
        self.is_game_over = False

        # Wind strength for each column (0-indexed)
        self.wind = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]

        self.actions = {'U': (-1, 0), 'D': (1, 0), 'L': (0, -1), 'R': (0, 1)}
        self.all_possible_actions = ('U', 'D', 'L', 'R')

    def reset(self):
        self.current_state = self.start_state
        self.is_game_over = False
        return self.current_state

    def is_terminal(self, s):
        return s == self.goal_state

    def game_over(self):
        return self.is_game_over

    def get_state(self):
        return self.current_state

    def move(self, action):
        if self.is_game_over:
            return 0.0

        r, c = self.current_state
        dr, dc = self.actions[action]

        # Apply action
        r_new, c_new = r + dr, c + dc
        # Apply wind
        wind_strength = self.wind[c_new] if 0 <= c_new < self.cols else 0
        r_new -= wind_strength

        # Boundary checks
        r_new = max(0, min(self.rows - 1, r_new))
        c_new = max(0, min(self.cols - 1, c_new))
        self.current_state = (r_new, c_new)

        # Reward
        if self.current_state == self.goal_state:
            self.is_game_over = True
            return 0.0
        else:
            return -1.0

    def all_states(self):
        return [(r, c) for r in range(self.rows) for c in range(self.cols)]


# --- 2. EXPLORATION and HELPERS ---

ALL_POSSIBLE_ACTIONS = ('U', 'D', 'L', 'R')


def max_dict(d):
    if not d:
        return np.random.choice(ALL_POSSIBLE_ACTIONS), 0.0
    max_val = max(d.values())
    max_keys = [k for k, v in d.items() if v == max_val]
    return np.random.choice(max_keys), max_val


def epsilon_greedy(Q, s, eps):
    """
    Performs epsilon-greedy action selection.
    """
    # Should select a random action with probability 'eps'
    # and the greedy action (using max_dict) with probability '1 - eps'.
    # Handle the case where state 's' is not yet in Q or Q[s] is empty.
    if np.random.random() < eps:
        return np.random.choice(ALL_POSSIBLE_ACTIONS)
    else:
        q_values =  Q.get(s, {})
        if not q_values:
            return np.random.choice(ALL_POSSIBLE_ACTIONS)
        else:
            return max_dict(Q[s])[0]



def ucb_exploration(Q, N, s, c, t):
    """
    Performs UCB (Upper Confidence Bound) action selection.
    Q: action-value function dictionary
    N: visit count dictionary for (state, action) pairs
    s: current state
    c: exploration constant
    t: current time step (or total step count)
    """
    if s not in Q or not Q[s]: return np.random.choice(ALL_POSSIBLE_ACTIONS)

    # Select unvisited actions if they exist (N(s, a) = 0).
    unvisited = [a for a in ALL_POSSIBLE_ACTIONS if N.get((s, a), 0) == 0]
    if unvisited:
        return np.random.choice(unvisited)

    # Otherwise, compute UCB value: Q(s, a) + c * sqrt(log(t) / N(s, a))
    ucb_vals={}
    for a in ALL_POSSIBLE_ACTIONS:
        n_value = N[(s,a)]
        ucb_vals[a] = Q[s][a] + c * np.sqrt(np.log(t)/n_value  )

    # Select action with max UCB value.
    max_ucb = max(ucb_vals.values())
    max_actions = [a for a, v in ucb_vals.items() if v == max_ucb]
    return np.random.choice(max_actions)
# --- 3. UPDATE RULES ---

def sarsa_update(Q, s, a, r, s2, a2, gamma, alpha, grid):
    """
    Performs SARSA update.
    """
    # Remember Q(s2, a2) is 0 if s2 is terminal (goal state).

    if grid.game_over():
        q_s2_a2 = 0.0
    else:
        if grid.is_terminal(s2):
            q_s2_a2 =0.0
        else:
            q_s2_a2 = Q[s2][a2]
        # s = current state , a = current action , s2 = next state , a2 = next action, q_s2_a2 = next state q value , r = reward ,
    Q[s][a] = Q[s][a] + alpha * (r + gamma * q_s2_a2 - Q[s][a])
    return Q


def q_learning_update(Q, s, a, r, s2, _a2, gamma, alpha, grid):
    """
    Performs q_learning update.
    """
    max_q_s2 = 0.0  # Placeholder

    if grid.game_over():
        max_q_s2 = 0.0
    else:
        _, max_q_s2 = max_dict(Q[s2])  # Q learning so max

    Q[s][a] = Q[s][a] + alpha * (r + gamma * max_q_s2 - Q[s][a])

    return Q


def expected_q_learning_update(Q, s, a, r, s2, a2, gamma, alpha, grid, eps=0.2):
    """
    Performs Expected SARSA (Expected Q-Learning) update.
    Q(s, a) <- Q(s, a) + alpha * [r + gamma * E[Q(s2, A)] - Q(s, a)]
    where E[Q(s2, A)] is the expected value under the epsilon-greedy policy:

    Let 'n' be the number of possible actions.
    Let 'a*' be the greedy action for state s2: a* = argmax_a Q(s2, a)

    The policy pi(a | s2) is defined as:
    pi(a* | s2) = pi_star = 1 - eps + eps / n
    pi(a | s2) = pi_other = eps / n   (for a != a*)

    E[Q(s2, A)] = sum_a pi(a | s2) * Q(s2, a)
    """
    #print("EXPECTED Q LEARNING")
    n = len(ALL_POSSIBLE_ACTIONS)
    if grid.game_over():
        expected_q_s2 = 0.0
    elif grid.is_terminal(s2):
        expected_q_s2 = 0.0
    else:
        expected_q_s2 = 0.0
        # Compute expected_q_s2

        a_star, _ = max_dict(Q[s2])
        for action in ALL_POSSIBLE_ACTIONS:
            if action == a_star:
                pi_star =  1 - eps + eps / n
                expected_q_s2 += pi_star*Q[s2][a]
            else:
                pi_other =eps / n
                expected_q_s2 += pi_other*Q[s2][a]

    # Now perform the update below

    Q[s][a] =  Q[s][a] + alpha * (r + gamma * expected_q_s2 - Q[s][a])

    return Q


def double_q_learning_update(QA, QB, s, a, r, s2, a2, gamma, alpha, grid):
    """
    Performs Double Q-Learning update.
    Randomly chooses QA or QB to update and uses the other for target selection.

    Let Q_upd be the Q-table chosen for the update (either QA or QB).
    Let Q_sel be the other Q-table used for value selection.

    The update rule for Q_upd is:
    Q_upd(s, a) <- Q_upd(s, a) + alpha * [r + gamma * Q_sel(s2, argmax_a' Q_upd(s2, a')) - Q_upd(s, a)]

    If s2 is terminal, the target is simply r.
    """
    # TODO: Implement double_q_learning_update function
    # Randomly choose Q_upd (the one to update) and Q_sel (the one for selection).

    Q_upd, Q_sel = None, None  # Initialize Q_upd, Q_sel correctly here)
    updated_is_QA = None  # Initialize a flag to check if QA was updated or QB
    if np.random.rand() < 0.5:
        # TODO : Write code here
        print('Kindly implement the random selection of Q_upd and Q_sel')
    else:
        # TODO : Write code here
        print('Kindly implement the random selection of Q_upd and Q_sel')

    # Calculate the target (Read the docstring carefully)
    target = 0.0  # TODO : Placeholder, Implement correctly here

    # TODO : write the update rule here

    # TODO : return the correct (QA, QB) tuple based on which was updated
    raise NotImplementedError("double_q_learning_update not fully implemented. Student must complete the update rule.")


# --- 4. CONTROL LOOP ---
# --- 4. CONTROL LOOP ---
def run_control(grid, Q_init, N_init, n_episodes, alpha, gamma,
                algorithm_name, exploration_type, **kwargs):
    np.random.seed(0)

    # Initialize Q-tables
    ALL_STATES = [s for s in grid.all_states() if not grid.is_terminal(s)]

    # Get all states from the grid environment (except the terminal/goal state)
    Q1 = {s: {a: Q_init.get((s, a), 0.0) for a in ALL_POSSIBLE_ACTIONS} for s in ALL_STATES}
    Q2 = None

    is_double_q = (algorithm_name == "Double Q-Learning")
    if is_double_q:
        # Initialize Q2 to all 0.0 values for all states and actions
        Q2 = {s: {a: 0.0 for a in ALL_POSSIBLE_ACTIONS} for s in ALL_STATES}


    N = N_init.copy()
    # Function to choose the appropriate update function DO NOT CHANGE
    update_fn = {
        'SARSA': sarsa_update,
        'Q-Learning': q_learning_update,
        'Expected Q-Learning': expected_q_learning_update,
        'Double Q-Learning': double_q_learning_update
    }[algorithm_name]

    # ---------DO NOT CHANGE--------
    eps = kwargs.get('eps', 0.2)
    c = kwargs.get('c', 2)
    t = 1  # Total time step counter for UCB

    # ---------DO NOT CHANGE-----

    # Helper function to get the primary Q-table for exploration
    def get_exploration_Q():
        return Q1

    # Helper function for action selection
    def select_action(state, current_t):
        Q_explore = get_exploration_Q()
        #  Implement action selection based on exploration_type
        #  USe the exploration functions defined earlier
        if exploration_type == "Epsilon-greedy":
            return epsilon_greedy(Q_explore, state, eps)
        elif exploration_type == "UCB":
            return ucb_exploration(Q_explore, N, state, c, current_t)
        return None

    # --- Main Control Loop ---
    for episode in range(n_episodes):
        s = grid.reset()
        a = select_action(s, t)  # Initial action selection
        print("EPISODE NUMBER : {}",episode)
       # loops = 0
        while not grid.game_over():
            r = grid.move(a)
            s2 = grid.get_state()
            # Select the next action (a2)
            a2 = select_action(s2, t)

            t += 1
            N[(s, a)] = N.get((s, a), 0) + 1

            # --- Q-Table Update ---
            if is_double_q:
                # Double Q-Learning update
                #  Call the appropriate update function and update Q1, Q2
                Q1, Q2 = update_fn(Q1, Q2, s, a, r, s2, a2, gamma, alpha, grid)
            else:
                # Single Q-Table Algorithm update
                if algorithm_name == "Expected Q-Learning":

                    Q1 =  update_fn(Q1, s, a, r, s2, a2, gamma, alpha, grid, eps=eps)
                else:
                    # SARSA or Q-Learning update
                    Q1 =  update_fn(Q1, s, a, r, s2, a2, gamma, alpha, grid)

            s = s2
            a = a2

    #  Return the correct Q-table(s) based on method used {single or double Q-learning}
    if is_double_q:
        return Q1, Q2
    else:
        return Q1

# --- 5. PATH TRACING and PLOTTING ---

def trace_optimal_path(Q, grid):
    s = grid.start_state
    path = [s]
    steps = 0
    max_steps = grid.rows * grid.cols * 10

    while s != grid.goal_state and steps < max_steps:
        a = max_dict(Q.get(s, {}))[0]
        r, c = s
        dr, dc = grid.actions[a]
        r_new, c_new = r + dr, c + dc
        wind = grid.wind[c_new] if 0 <= c_new < grid.cols else 0
        r_new -= wind
        r_new = max(0, min(grid.rows - 1, r_new))
        c_new = max(0, min(grid.cols - 1, c_new))
        s_next = (r_new, c_new)
        if (s_next == s and steps > 0) or path.count(s_next) > 1:
            break
        path.append(s_next)
        s = s_next
        steps += 1

    return path, steps


def plot_path(grid, path, title, ax):
    ax.set_xlim(-0.5, grid.cols - 0.5)
    ax.set_ylim(grid.rows - 0.5, -0.5)
    ax.set_xticks(np.arange(-0.5, grid.cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, grid.rows, 1), minor=True)
    ax.grid(which='minor', color='black', linestyle='-', linewidth=1)
    ax.tick_params(which='minor', size=0)
    ax.tick_params(labelbottom=False, labelleft=False)

    ax.text(grid.start_state[1], grid.start_state[0], 'S',
            ha='center', va='center', color='green', fontsize=16, weight='bold')
    ax.text(grid.goal_state[1], grid.goal_state[0], 'G',
            ha='center', va='center', color='red', fontsize=16, weight='bold')

    for c, w in enumerate(grid.wind):
        if w > 0:
            color = 'lightblue' if w == 1 else 'skyblue'
            ax.add_patch(plt.Rectangle((c - 0.5, -0.5), 1, grid.rows,
                                       color=color, alpha=0.3))

    coords = np.array(path)
    if coords.shape[0] > 1 and path[-1] == grid.goal_state:
        ax.plot(coords[:, 1], coords[:, 0], 'o-', color='purple',
                markersize=5, linewidth=2)
        ax.set_title(title, fontsize=10)
    elif coords.shape[0] > 1:
        ax.plot(coords[:, 1], coords[:, 0], 'o--', color='orange',
                markersize=5, linewidth=2)
        ax.set_title(f"{title}\n(Path terminated, Goal NOT reached)", fontsize=10)
    else:
        ax.set_title(f"{title}\n(Path not found)", fontsize=10)


# --- 6. MAIN EXECUTION ---

def run_all_experiments():
    N_EPISODES = 4000
    ALPHA = 0.5
    ALPHA_DOUBLE_Q = 0.997
    GAMMA = 1.0
    EPSILON = 0.2
    UCB_C = 2

    grid = WindyGrid()
    results = {}

    experiments = [
        #("a) SARSA, ε-greedy", 'SARSA', 'Epsilon-greedy', {'eps': EPSILON}),
        #("b) SARSA, UCB", 'SARSA', 'UCB', {'c': UCB_C}),
        #("c) Q-Learning, ε-greedy", 'Q-Learning', 'Epsilon-greedy', {'eps': EPSILON}),
        #("d) Q-Learning, UCB", 'Q-Learning', 'UCB', {'c': UCB_C}),
        ("e) Exp. Q-Learning, ε-greedy", 'Expected Q-Learning', 'Epsilon-greedy', {'eps': EPSILON}),
        #("f) Exp. Q-Learning, UCB", 'Expected Q-Learning', 'UCB', {'c': UCB_C}),
        #("g) Double Q-Learning, ε-greedy", 'Double Q-Learning', 'Epsilon-greedy', {'eps': EPSILON}),
        #("h) Double Q-Learning, UCB", 'Double Q-Learning', 'UCB', {'c': UCB_C}),
    ]

    Q_init = {}
    N_init = {}

    for name, alg, exp_type, params in experiments:
        print(f"Running: {name}...")
        Q_final = run_control(
            grid=grid,
            Q_init=Q_init,
            N_init=N_init,
            n_episodes=N_EPISODES,
            alpha=ALPHA,
            gamma=GAMMA,
            algorithm_name=alg,
            exploration_type=exp_type,
            **params
        )
        if isinstance(Q_final, tuple):
            Q1, Q2 = Q_final
            Q_for_path = {
                s: {a: 0.5 * (Q1[s][a] + Q2[s][a]) for a in ALL_POSSIBLE_ACTIONS}
                for s in Q1
            }
        else:
            Q_for_path = Q_final

        path, steps = trace_optimal_path(Q_for_path, grid)
        results[name] = {'path': path, 'steps': steps}

    fig, axes = plt.subplots(4, 2, figsize=(15, 20))
    axes = axes.flatten()
    for i, (name, data) in enumerate(results.items()):
        plot_path(grid, data['path'], f"{name}\nSteps: {data['steps']}", axes[i])

    fig.suptitle(f"Windy Gridworld Shortest Learnt Path Comparison ({N_EPISODES} Episodes)", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.show()

    print("\n" + "=" * 50)
    print("Shortest Path Results After 500 Episodes")
    print("=" * 50)
    print(f"{'Algorithm':<30} | {'Steps in Shortest Learnt Path':>30}")
    print("-" * 63)
    for name, data in results.items():
        print(f"{name:<30} | {data['steps']:>30}")


if __name__ == '__main__':
    run_all_experiments()
