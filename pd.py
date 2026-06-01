import matplotlib.pyplot as plt
import numpy as np
from numpy import pi

from controllers.pd_controller import PDDecentralizedController
from trajectory_generators.sinusonidal import Sinusoidal
from utils.simulation import simulate

Tp = 0.001
end = 5

traj_gen = Sinusoidal(np.array([0., 1.]), np.array([2., 2.]), np.array([0., 0.]))

kp_real = np.array([80.0, 80.0])
kd_real = np.array([20.0, 20.0])

controller = PDDecentralizedController(kp_real, kd_real)

Q, Q_d, u, T, _ = simulate("PYBULLET", traj_gen, controller, Tp, end, multimodel=True)

plt.figure("Klasyczny regulator PD - Zadanie 5 i 6")

plt.subplot(311)
plt.plot(T, Q_d[:, 0], 'b--', label='Zadana qd_1')
plt.plot(T, Q[:, 0], 'r', label='Rzeczywista q_1')
plt.title("Pozycja Przegubu 1")
plt.legend()
plt.grid(True)

plt.subplot(312)
plt.plot(T, Q_d[:, 1], 'b--', label='Zadana qd_2')
plt.plot(T, Q[:, 1], 'r', label='Rzeczywista q_2')
plt.title("Pozycja Przegubu 2")
plt.legend()
plt.grid(True)

plt.subplot(313)
plt.plot(T, u[:, 0], 'g', label='Sterowanie u_1')
plt.plot(T, u[:, 1], 'm', label='Sterowanie u_2')
plt.title("Sygnały sterujące")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()