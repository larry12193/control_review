import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# Point mass (kg)
m = 1.0
# Pendulum length (m)
l = 0.25
# Gravity (m/s^2)
g = 9.8
# Damping (N-s/m)
b = 1.0
# Control input limit (N-m)
u_lim = 2.4
# Vertical stability gain
k = 2.0
# Setup time vector (s)
t = np.arange(0,60,1e-3)
# Initial state of pendulum (th,dth)
q0 = [0.05,0]
# Global container for control policy
u_global = np.array([[0],[0]])

# Swingup controller
def f_swingup(x,t):
    global u_global

    # Switching manifold
    if np.absolute(x[0]) > np.pi-0.3:
        # Simple negation of gravity based control
        u = 2*m*g*l*np.sin(x[0])
    else:
        # Energy shaping
        Edes = m*g*l
        Ecur = 0.5*m*l**2*x[1]**2 - m*g*l*np.cos(x[0])
        Eerr = Ecur - Edes
        u = -k*x[1]*Eerr

    # Apply torque limit
    if np.absolute(u) > u_lim:
        u = np.sign(u)*u_lim

    u_global = np.append(u_global,np.array([[t],[u]]),1)

    dx_dt = [0.0,0.0]

    # Compute model update
    dx_dt[0] = x[1]
    dx_dt[1] = -m*g*l*np.sin(x[0]) - b*x[1] + u
    return dx_dt

# Simulate ODE
s = odeint(f_swingup,y0=q0,t=t)
print(u_global[0,:])
# Plot response
plt.figure(1)
plt.plot(t,s[:,0],'b',linewidth=3,label='Angular Position')
plt.plot(t,np.pi*np.ones((len(s[:,0]),1),dtype=np.float),'k--',label='Target')
plt.xlabel('Time (s)')
plt.ylabel('Pendulum Position (rad)')
plt.title('Pendulum Energy Shaping')
plt.grid('on')
plt.legend()

# Plot control usage
plt.figure(2)
plt.plot(u_global[0,:],u_global[1,:],'b',linewidth=3,label='Control Policy')
plt.xlabel('Time (s)')
plt.ylabel('Torque Command')
plt.title('Pendulum Control Usage')
plt.grid('on')
plt.legend()

# Discretize state space
q1 = np.linspace(-2*np.pi,2*np.pi,dtype=np.float)
q2 = np.linspace(-2*m*g*l/b,2*m*g*l/b,dtype=np.float)
Q1,Q2 = np.meshgrid(q1,q2)
U,V = np.zeros(Q1.shape), np.zeros(Q2.shape)
# Compute phase space
for i in range(Q1.shape[0]):
    for j in range(Q1.shape[1]):
        th = Q1[i,j]
        dth = Q2[i,j]
        dq = f_swingup([th,dth],0)
        U[i,j] = dq[0]
        V[i,j] = dq[1]

plt.figure(3)
plt.quiver(Q1,Q2,U,V,color='r')
plt.plot(s[:,0],s[:,1],color='b')
plt.plot([s[0,0]],[s[0,1]],color='g')
plt.plot([s[-1,0]],[s[-1,1]],color='k')
plt.xlabel('$q$')
plt.ylabel('$\dot{q}$')
plt.title('Control+Plant Phase Portrait')
plt.show()
