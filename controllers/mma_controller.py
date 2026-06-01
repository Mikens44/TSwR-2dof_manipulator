import numpy as np
from .controller import Controller
from models.manipulator_model import ManiuplatorModel


class MMAController(Controller):
    def __init__(self, Tp):
        self.Tp = Tp
        # TODO: Fill the list self.models with 3 models of 2DOF manipulators with different m3 and r3
        # I:   m3=0.1,  r3=0.05
        # II:  m3=0.01, r3=0.01
        # III: m3=1.0,  r3=0.3
        self.models = [
            ManiuplatorModel(Tp, m3=0.1, r3=0.05), 
            ManiuplatorModel(Tp, m3=0.01, r3=0.01), 
            ManiuplatorModel(Tp, m3=1.0, r3=0.3)
        ]
        self.i = 0

        self.K_p = 225.0
        self.K_d = 30.0

        self.last_u = np.zeros(2)
        self.last_x = None

        self.model_history = []
        self.error_history = []
        self.object_history = []

    def choose_model(self, x):
        if self.last_x is None:
            self.last_x = x.copy()
            self.model_history.append(self.i)
            self.error_history.append([0.0, 0.0, 0.0])
            return
        
        q_prev = self.last_x[:2]
        q_dot_prev = self.last_x[2:]
        
        errors = []
        
        tau_prev = self.last_u[:, np.newaxis]
        q_dot_prev_col = q_dot_prev[:, np.newaxis]
        
        for idx, model in enumerate(self.models):
            M = model.M(self.last_x)
            C = model.C(self.last_x)

            q_ddot_pred = np.linalg.inv(M) @ (tau_prev - C @ q_dot_prev_col)

            q_dot_pred = q_dot_prev_col + self.Tp * q_ddot_pred
            q_pred = q_prev[:, np.newaxis] + self.Tp * q_dot_pred 

            x_pred = np.vstack((q_pred, q_dot_pred)).flatten()
            
            error = np.linalg.norm(x - x_pred)
            errors.append(error)
            print(f"Model {idx} (m3={model.m3}): błąd = {error:.6f}")
        
        self.i = np.argmin(errors)
        print(f"--> WYBRANO MODEL: {self.i}")
        self.model_history.append(self.i)
        self.error_history.append(errors.copy())
        
        self.last_x = x.copy()

    def calculate_control(self, x, q_r, q_r_dot, q_r_ddot):
        self.choose_model(x)
        q = x[:2]
        q_dot = x[2:]
        e = q_r - q
        e_dot = q_r_dot - q_dot
        v = q_r_ddot + self.K_d * e_dot + self.K_p * e
        M = self.models[self.i].M(x)
        C = self.models[self.i].C(x)
        u = M @ v[:, np.newaxis] + C @ q_dot[:, np.newaxis]
        u = u.flatten()
        self.last_u = u.copy()
        return u
