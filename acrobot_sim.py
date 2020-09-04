from numpy import sin, cos
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import matplotlib.animation as animation

class Acrobot:

    def __init__(self,
                 init_state = [np.pi*120./180., 0., np.pi*-20./180., 0.],
                 L1=1.0,
                 LC1=0.5,
                 L2=1.0,
                 LC2=0.5,
                 M1=1.0,
                 M2=1.0,
                 G=9.8,
                 damp_ratio = 0.5,
                 origin=(0, 0)):

        # Save initial state
        self.init_state = np.asarray(init_state, dtype='float')
        self.origin = origin
        self.time_elapsed = 0

        # Compute moment of inertia for each link
        I1 = (M1*L1**2)/3.0
        I2 = (M2*L2**2)/3.0

        # Save relevant parameters
        self.params = (L1, LC1, L2, LC2, M1, I1, M2, I2, G)
        self.damp_ratio = damp_ratio

        # Set state of acrobot to initial state
        self.state = self.init_state

    def position(self):
        # Extract parameters
        (L1, LC1, L2, LC2, M1, I1, M2, I2, G) = self.params

        # Compute x and y positions of each part of the link
        x = np.array([self.origin[0],
                       L1 * cos(self.state[0]),
                       L1 * cos(self.state[0]) + L2 * cos(self.state[0]+self.state[2])])
        y = np.array([self.origin[1],
                       L1 * sin(self.state[0]),
                       L1 * sin(self.state[0]) + L2 * sin(self.state[0]+self.state[2])])
        return (x, y)

    def energy(self):
        # Extract parameters
        (L1, LC1, L2, LC2, M1, I1, M2, I2, G) = self.params

        q1 = self.state[0]
        dq1 = self.state[1]
        q2 = self.state[2]
        dq2 = self.state[3]

        c1 = cos(q1)
        c2 = cos(q2)
        s1 = sin(q1)
        s2 = sin(q2)
        s12 = sin(q1+q2)
        dq1_2 = dq1**2
        dq2_2 = dq2**2

        K = (0.5*I1*dq1_2 + 0.5*I2*(dq1_2+dq2_2)
           + 0.5*M1*(LC1*s1**2*dq1_2 +LC1*c1**2*dq1_2)
           + 0.5*M2*(((-L1*s1*dq1 + LC2*(s1*c2+s2*c1)*(-dq1-dq2)))**2
           + (L1*c1*dq1 + LC2*(-s1*s2+c1*c2)*(dq1+dq2))**2))
        U = M1*G*LC1*s1 + M2*G*(L1*s1 + LC2*s12)

        return (U,K)

    def dstate_dt(self, state, t):

        (L1, LC1, L2, LC2, M1, I1, M2, I2, G) = self.params

        q1 = state[0]
        dq1 = state[1]
        q2 = state[2]
        dq2 = state[3]

        dydx = np.zeros_like(state)
        dydx[0] = dq1
        dydx[2] = dq2

        # Mass matrix
        m11 = I1 + I2 + M2*L1**2 + 2.0*M2*L1*LC2*cos(q2) + M1*LC1**2 + M2*LC2**2
        m12 = I2 + M2*L1*LC2*cos(q2) + M2*LC2**2
        m21 = I2 + M2*L1*LC2*cos(q2) + M2*LC2**2
        m22 = I2 + M2*LC2**2
        M = np.array([[m11,m12],[m21,m22]])
        Minv = np.linalg.inv(M)

        # Corriolis matrix
        c11 = -2*M2*L1*LC2*sin(q2)*dq2
        c12 = -M2*L1*LC2*sin(q2)*dq2
        c21 = M2*L1*LC2*sin(q2)*dq1
        c22 = 0
        C = np.array([[c11,c12],[c21,c22]])
        Cq = np.matmul(C,np.array([[dq1],[dq2]]))

        # Gravitational matrix
        g11 = M2*G*L1*cos(q1) + M1*G*LC1*cos(q1) + M2*G*LC2*cos(q1+q2)
        g21 = M2*G*LC2*cos(q1+q2)
        G = np.array([[g11],[g21]])

        # Damping
        D = self.damp_ratio*np.array([[dq1],[dq2]])

        # Control
        B = np.array([[0.0],[1.0]])
        Bu = B*0.0

        # Compute dynamics
        xdot = np.matmul(Minv,Bu-Cq-G-D)

        dydx[1] = xdot[0]
        dydx[3] = xdot[1]

        return dydx

    def step(self, dt):
        """execute one time step of length dt and update state"""
        self.state = integrate.odeint(self.dstate_dt, self.state, [0, dt])[1]
        self.time_elapsed += dt

#------------------------------------------------------------
# set up initial state and global variables
pendulum = Acrobot([0.0, 0.0, 0., 0.0])
dt = 1./30 # 30 fps

#------------------------------------------------------------
# set up figure and animation
fig = plt.figure()
ax = fig.add_subplot(111, aspect='equal', autoscale_on=False,
                     xlim=(-2, 2), ylim=(-2, 2))
ax.grid()

line1, = ax.plot([], [], 'o-', lw=2)
line2, = ax.plot([], [], 'o-',lw=2)
time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
energy_text = ax.text(0.02, 0.90, '', transform=ax.transAxes)

def init():
    """initialize animation"""
    line1.set_data([], [])
    line2.set_data([], [])
    time_text.set_text('')
    energy_text.set_text('')
    return line1, line2, time_text, energy_text

def animate(i):
    """perform animation step"""
    global pendulum, dt
    pendulum.step(dt)
    x,y = pendulum.position()

    line1.set_data([pendulum.origin[0],x[1]],[pendulum.origin[1],y[1]])
    line2.set_data([x[1],x[2]],[y[1],y[2]])

    time_text.set_text('time = %.1f' % pendulum.time_elapsed)
    energy_text.set_text('energy = (U%.3f,K%.3f) J' % pendulum.energy())
    return line1, line2, time_text, energy_text

# choose the interval based on dt and the time to animate one step
from time import time
t0 = time()
animate(0)
t1 = time()
interval = 1000 * dt - (t1 - t0)

ani = animation.FuncAnimation(fig, animate, frames=300,
                              interval=interval, blit=True, init_func=init)
plt.show()
