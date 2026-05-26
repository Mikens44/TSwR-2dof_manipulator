from copy import copy
import numpy as np


class ESO:
    def __init__(self, A, B, W, L, state, Tp):
        self.A = A
        self.B = B
        self.W = W
        self.L = L
        self.state = np.pad(np.array(state), (0, A.shape[0] - len(state)))
        self.Tp = Tp
        self.states = []

    def set_B(self, B):
        self.B = B

    def update(self, q, u):
        self.states.append(copy(self.state))
        ### TODO implement ESO update
        z1_pred = self.state[0]

        u_col = np.array([[u]]) if np.isscalar(u) else np.array(u)[:, np.newaxis]
        z_col = self.state[:, np.newaxis] if len(self.state.shape) == 1 else self.state

        dot_f = np.array([[0.0]])

        dot_z = self.A @ z_col + self.B @ u_col + self.W @ dot_f + self.L * (q - z1_pred)

        z_new = z_col + self.Tp * dot_z

        self.state = z_new.flatten()

    def get_state(self):
        return self.state
