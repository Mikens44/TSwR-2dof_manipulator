import numpy as np
from .adrc_joint_controller import ADRCJointController
from .controller import Controller
from models.manipulator_model import ManiuplatorModel


class ADRController(Controller):
    def __init__(self, Tp, params, adaptive_b=False):
        self.joint_controllers = []
        for param in params:
            self.joint_controllers.append(ADRCJointController(*param, Tp))
        self.adaptive_b = adaptive_b
        self.model = ManiuplatorModel(Tp)

    def update_b(self, x):
        M = self.model.M(x)
        M_inv = np.linalg.inv(M)

        self.joint_controllers[0].set_b(M_inv[0, 0])
        self.joint_controllers[1].set_b(M_inv[1, 1])

    def calculate_control(self, x, q_d, q_d_dot, q_d_ddot):
        if self.adaptive_b:
            self.update_b(x)

        u = []

        for i, controller in enumerate(self.joint_controllers):
            u.append(
                controller.calculate_control(
                    [x[i], x[i + 2]],
                    q_d[i],
                    q_d_dot[i],
                    q_d_ddot[i]
                )
            )

        return np.array(u)

