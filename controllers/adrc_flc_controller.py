import numpy as np

from models.manipulator_model import ManiuplatorModel
from observers.eso_mimo import ESOMIMO
from .controller import Controller


class ADRFLController(Controller):
    def __init__(self, Tp, q0, Kp, Kd, p):
        self.model = ManiuplatorModel(Tp)

        self.Kp = Kp
        self.Kd = Kd
        self.Tp = Tp

        p = np.asarray(p, dtype=float).reshape(2)

        self.L = np.zeros((6, 2))

        self.L[0:2, :] = np.diag(3 * p)
        self.L[2:4, :] = np.diag(3 * p**2)
        self.L[4:6, :] = np.diag(p**3)

        W = np.zeros((2, 6))
        W[:, 0:2] = np.eye(2)

        A = np.zeros((6, 6))
        B = np.zeros((6, 2))

        z0 = np.zeros(6)
        z0[0:2] = q0[0:2]
        z0[2:4] = q0[2:4]
        z0[4:6] = 0.0

        self.eso = ESOMIMO(A, B, W, self.L, z0, Tp)

        self.u_prev = np.zeros(2)
        self.eso_history = []

        self.update_params(q0[0:2], q0[2:4])

    def update_params(self, q, q_dot):
        q = np.asarray(q, dtype=float).reshape(2)
        q_dot = np.asarray(q_dot, dtype=float).reshape(2)

        x_state = np.concatenate([q, q_dot])

        M = self.model.M(x_state)
        C = self.model.C(x_state)

        M_inv = np.linalg.inv(M)

        A = np.zeros((6, 6))

        A[0:2, 2:4] = np.eye(2)
        A[2:4, 2:4] = -M_inv @ C
        A[2:4, 4:6] = np.eye(2)

        B = np.zeros((6, 2))
        B[2:4, :] = M_inv

        self.eso.A = A
        self.eso.B = B

    def calculate_control(self, x, q_d, q_d_dot, q_d_ddot):
        x = np.asarray(x, dtype=float).reshape(4)

        q = x[0:2]
        q_dot = x[2:4]

        self.update_params(q, q_dot)

        self.eso.update(q, self.u_prev)

        z = self.eso.get_state().reshape(6)

        q_est = z[0:2]
        q_dot_est = z[2:4]
        f_est = z[4:6]

        q_d = np.asarray(q_d, dtype=float).reshape(2)
        q_d_dot = np.asarray(q_d_dot, dtype=float).reshape(2)
        q_d_ddot = np.asarray(q_d_ddot, dtype=float).reshape(2)

        v = (
            q_d_ddot
            + self.Kp @ (q_d - q_est)
            + self.Kd @ (q_d_dot - q_dot_est)
        )

        x_est = np.concatenate([q_est, q_dot_est])

        M = self.model.M(x_est)
        C = self.model.C(x_est)

        u = M @ (v - f_est) + C @ q_dot_est

        self.u_prev = u.copy()
        self.eso_history.append(z.copy())

        return u
    