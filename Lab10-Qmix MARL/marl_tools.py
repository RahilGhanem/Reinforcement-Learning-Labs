"""
marl_tools.py  —  Helper module for the QMIX Lab

Provides:
  - TreasureHunt : a simple cooperative 5×5 grid-world environment
  - plot_learning_curves : smoothed return curves for multiple algorithms
  - render_episode : run and render one episode step-by-step
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import random
import time
from IPython import display

# ── Matplotlib defaults (match the template style) ──────────────────────────
plt.rc('font',   size=14)
plt.rc('axes',   titlesize=13)
plt.rc('axes',   labelsize=12)
plt.rc('xtick',  labelsize=10)
plt.rc('ytick',  labelsize=10)
plt.rc('legend', fontsize=11)
plt.rc('figure', titlesize=14)


# ═══════════════════════════════════════════════════════════════════════════════
#  TreasureHunt environment
# ═══════════════════════════════════════════════════════════════════════════════

class TreasureHunt:
    """
    A 5×5 cooperative grid-world for 2 agents.

    Layout (initial):
        A  .  .  .  T0
        .  .  .  .  .
        .  .  .  .  .
        .  .  .  .  .
        T1 .  .  .  B

        A  = Agent 0  starts at (0, 0)  — must collect T0 at (0, 4)
        B  = Agent 1  starts at (4, 4)  — must collect T1 at (4, 0)

    Rewards
    -------
    +1.0  when an agent steps onto its own treasure (once per treasure)
    -0.05 per time-step (encourages efficient navigation)

    Episode ends when both treasures are collected OR max_steps is reached.

    Partial observations (per agent, 5 values)
    ------------------------------------------
    [own_row/4,  own_col/4,  goal_row/4,  goal_col/4,  collected?]
    The agent does NOT see the other agent's position.

    Global state (6 values, used ONLY during centralised training)
    ---------------------------------------------------------------
    [a0_row/4, a0_col/4, a1_row/4, a1_col/4, collected_0, collected_1]
    """

    # ── grid constants ───────────────────────────────────────────────────────
    SIZE         = 5
    N_AGENTS     = 2
    N_ACTIONS    = 4          # UP, DOWN, LEFT, RIGHT
    OBS_SIZE     = 5
    STATE_SIZE   = 6

    TREASURE_POS = [(0, 4), (4, 0)]   # T0, T1  (fixed)
    START_POS    = [(0, 0), (4, 4)]   # Agent 0, Agent 1 (fixed)

    # action → (Δrow, Δcol)
    _DELTA = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
    ACTION_NAMES = ['UP', 'DOWN', 'LEFT', 'RIGHT']

    # ── constructor ──────────────────────────────────────────────────────────
    def __init__(self, max_steps=40, step_penalty=0.05):
        self.max_steps   = max_steps
        self.step_penalty = step_penalty
        self.reset()

    # ── reset / step ─────────────────────────────────────────────────────────
    def reset(self):
        """
        Reset the episode.

        Returns
        -------
        obs : list of 2 numpy arrays, each of shape (OBS_SIZE,)
        """
        self._t          = 0
        self._agent_pos  = list(self.START_POS)       # [(row,col), ...]
        self._collected  = [False, False]
        return self._get_obs()

    def step(self, actions):
        """
        Advance the environment by one time-step.

        Parameters
        ----------
        actions : list of 2 ints (one action per agent)

        Returns
        -------
        obs      : list of 2 numpy arrays (next observations)
        reward   : float (shared team reward)
        done     : bool
        info     : dict with key 'state' → numpy array of shape (STATE_SIZE,)
        """
        self._t += 1
        reward   = -self.step_penalty

        for i, a in enumerate(actions):
            if not self._collected[i]:                        # frozen once collected
                dr, dc = self._DELTA[a]
                r,  c  = self._agent_pos[i]
                nr = max(0, min(self.SIZE - 1, r + dr))
                nc = max(0, min(self.SIZE - 1, c + dc))
                self._agent_pos[i] = (nr, nc)

                if self._agent_pos[i] == self.TREASURE_POS[i]:
                    self._collected[i] = True
                    reward += 1.0

        done  = all(self._collected) or self._t >= self.max_steps
        obs   = self._get_obs()
        state = self.get_state()
        return obs, reward, done, {'state': state}

    # ── observations / state ─────────────────────────────────────────────────
    def _get_obs(self):
        """Return list of per-agent observations."""
        return [self.get_observation(i) for i in range(self.N_AGENTS)]

    def get_observation(self, agent_id):
        """
        Compute the partial observation for agent `agent_id`.

        Returns a numpy array of shape (OBS_SIZE,) = (5,):
            [own_row/4, own_col/4, goal_row/4, goal_col/4, collected?]
        """
        G  = self.SIZE - 1
        r,  c  = self._agent_pos[agent_id]
        tr, tc = self.TREASURE_POS[agent_id]
        return np.array([
            r  / G,
            c  / G,
            tr / G,
            tc / G,
            float(self._collected[agent_id])
        ], dtype=np.float32)

    def get_state(self):
        """
        Compute the global state (used only during centralised training).

        Returns a numpy array of shape (STATE_SIZE,) = (6,):
            [a0_row/4, a0_col/4, a1_row/4, a1_col/4, collected_0, collected_1]
        """
        G       = self.SIZE - 1
        r0, c0  = self._agent_pos[0]
        r1, c1  = self._agent_pos[1]
        return np.array([
            r0 / G, c0 / G,
            r1 / G, c1 / G,
            float(self._collected[0]),
            float(self._collected[1])
        ], dtype=np.float32)

    # ── rendering ─────────────────────────────────────────────────────────────
    def render(self):
        """Print a text representation of the current grid."""
        grid = [['  .' for _ in range(self.SIZE)] for _ in range(self.SIZE)]

        for i, (tr, tc) in enumerate(self.TREASURE_POS):
            if not self._collected[i]:
                grid[tr][tc] = f' T{i}'

        labels = [' A', ' B']
        for i, (r, c) in enumerate(self._agent_pos):
            grid[r][c] = labels[i]

        print(f"Step {self._t:2d} | collected={self._collected}")
        for row in grid:
            print(' '.join(row))
        print()

    def render_matplotlib(self, ax=None):
        """Draw the current grid state on a matplotlib axis."""
        if ax is None:
            _, ax = plt.subplots(figsize=(4, 4))

        ax.set_xlim(-0.5, self.SIZE - 0.5)
        ax.set_ylim(-0.5, self.SIZE - 0.5)
        ax.set_xticks(range(self.SIZE))
        ax.set_yticks(range(self.SIZE))
        ax.grid(True, linewidth=0.5, color='grey')
        ax.set_aspect('equal')
        ax.invert_yaxis()

        # Treasures
        for i, (tr, tc) in enumerate(self.TREASURE_POS):
            color = 'gold' if not self._collected[i] else 'lightgrey'
            ax.add_patch(mpatches.FancyBboxPatch(
                (tc - 0.4, tr - 0.4), 0.8, 0.8,
                boxstyle='round,pad=0.05', facecolor=color, edgecolor='k'))
            ax.text(tc, tr, f'T{i}', ha='center', va='center', fontsize=9, fontweight='bold')

        # Agents
        agent_colors = ['steelblue', 'tomato']
        agent_labels = ['A', 'B']
        for i, (r, c) in enumerate(self._agent_pos):
            circle = plt.Circle((c, r), 0.35, color=agent_colors[i], zorder=3)
            ax.add_patch(circle)
            ax.text(c, r, agent_labels[i], ha='center', va='center',
                    color='white', fontsize=11, fontweight='bold', zorder=4)

        ax.set_title(f'Step {self._t}  collected={self._collected}', fontsize=10)
        return ax


# ═══════════════════════════════════════════════════════════════════════════════
#  Plotting utilities
# ═══════════════════════════════════════════════════════════════════════════════

def plot_learning_curves(returns_dict, window=30, title='Learning Curves'):
    """
    Plot smoothed episode-return curves for one or more algorithms.

    Parameters
    ----------
    returns_dict : dict  { algorithm_name : list_of_episode_returns }
    window       : int   smoothing window size
    title        : str
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    colors  = ['steelblue', 'tomato', 'seagreen', 'darkorange', 'mediumpurple']

    for (name, returns), color in zip(returns_dict.items(), colors):
        eps = np.arange(len(returns))

        # Faded raw curve
        ax.plot(eps, returns, alpha=0.18, color=color)

        # Smoothed curve
        if len(returns) >= window:
            kernel   = np.ones(window) / window
            smoothed = np.convolve(returns, kernel, mode='valid')
            ax.plot(np.arange(window - 1, len(returns)), smoothed,
                    label=name, color=color, linewidth=2)

    ax.set_xlabel('Episode')
    ax.set_ylabel('Total Return')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def render_episode(env, get_actions_fn, title='Episode Playback', delay=0.3):
    """
    Run one episode and render it frame-by-frame in the notebook.

    Parameters
    ----------
    env            : TreasureHunt instance
    get_actions_fn : callable(obs_list) → actions_list
    title          : str
    delay          : float  seconds between frames
    """
    obs  = env.reset()
    done = False
    total_reward = 0.0

    fig, ax = plt.subplots(figsize=(4, 4))
    fig.suptitle(title)

    while not done:
        ax.cla()
        env.render_matplotlib(ax)
        display.clear_output(wait=True)
        display.display(fig)
        time.sleep(delay)

        actions              = get_actions_fn(obs)
        obs, r, done, _info  = env.step(actions)
        total_reward        += r

    # Final frame
    ax.cla()
    env.render_matplotlib(ax)
    display.clear_output(wait=True)
    display.display(fig)
    plt.close(fig)
    print(f'\nEpisode finished — total reward: {total_reward:.3f}')
