import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from scipy.integrate import odeint

c = 5

A = np.array([[0.0,1.0],[0.0,-c]])
B = np.array([[0.0],[1.0]])
C = np.array([1.0,0.0])
D = 0.0

sys = signal.StateSpace(A,B,C,D)
t,y = signal.impulse(sys)

plt.figure(1)
plt.plot(t,y,'b--',linewidth=3,label='Double Integrator')
plt.show()
