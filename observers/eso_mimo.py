from copy import copy
import numpy as np


class ESOMIMO:
    def __init__(self, A, B, W, L, state, Tp):
        self.A = np.asarray(A, dtype=float)
        self.B = np.asarray(B, dtype=float)
        self.W = np.asarray(W, dtype=float)
        self.L = np.asarray(L, dtype=float)
        self.state = np.asarray(state, dtype=float).reshape(-1)
        self.Tp = Tp
        self.states = []

    def update(self, q, u):
        self.states.append(copy(self.state))

        z = self.state.reshape(6, 1)
        q = np.asarray(q, dtype=float).reshape(2, 1)
        u = np.asarray(u, dtype=float).reshape(2, 1)

        error = q - self.W @ z

        dot_z = (
            self.A @ z
            + self.B @ u
            + self.L @ error
        )

        self.state = (z + self.Tp * dot_z).flatten()

    def get_state(self):
        return self.state
    