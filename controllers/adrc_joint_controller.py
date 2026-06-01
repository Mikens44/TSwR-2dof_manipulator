import numpy as np
from observers.eso import ESO
from .controller import Controller


class ADRCJointController(Controller):
    def __init__(self, b, kp, kd, p, q0, Tp):
        self.b = b
        self.kp = kp
        self.kd = kd
        self.u_prev = 0.0

        A = np.array([
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0]
        ])

        B = np.array([
            [0.0],
            [b],
            [0.0]
        ])

        L = np.array([
            [3.0 * p],
            [3.0 * (p**2)],
            [p**3]
        ])

        W = np.array([
            [0.0],
            [0.0],
            [1.0]
        ])

        self.eso = ESO(A, B, W, L, q0, Tp)

    def set_b(self, b):
        self.b = float(b)

        self.eso.set_B(np.array([
            [0.0],
            [self.b],
            [0.0]
        ]))

    def calculate_control(self, x, q_d, q_d_dot, q_d_ddot):
        q_real = x[0]

        self.eso.update(q_real, self.u_prev)

        z = self.eso.get_state()

        q_est = z[0]
        q_dot_est = z[1]
        f_est = z[2]

        e = q_d - q_est
        e_dot = q_d_dot - q_dot_est

        v = q_d_ddot + self.kd * e_dot + self.kp * e

        disturbance_rejection = True

        if disturbance_rejection:
            u = (v - f_est) / self.b
        else:
            u = v / self.b

        self.u_prev = u

        return u
