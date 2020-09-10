from numpy import sin, cos
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import matplotlib.animation as animation
import control

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
                 damp_ratio = 0.1,
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
        self.update_energy()
        self.update_task_space()

        self.Q = np.diag([10,10,1,1])
        self.R = 1
        self.last_u = 0.0

    def update_task_space(self):
        # Extract parameters
        (L1, LC1, L2, LC2, M1, I1, M2, I2, G) = self.params

        # Compute x and y positions of each part of the link
        x = np.array([self.origin[0],
                       L1 * cos(self.state[0]),
                       L1 * cos(self.state[0]) + L2 * cos(self.state[0]+self.state[2])])
        y = np.array([self.origin[1],
                       L1 * sin(self.state[0]),
                       L1 * sin(self.state[0]) + L2 * sin(self.state[0]+self.state[2])])
        self.current_position = (x, y)

    def update_energy(self):
        self.current_energy = self.compute_energy(self.state)

    def compute_energy(self,state):
        # Extract parameters
        (L1, LC1, L2, LC2, M1, I1, M2, I2, G) = self.params

        q1 = state[0]
        dq1 = state[1]
        q2 = state[2]
        dq2 = state[3]

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
        # Extract parameters
        (L1, LC1, L2, LC2, M1, I1, M2, I2, G) = self.params

        q1 = (state[0]+np.pi) % (2*np.pi) - np.pi
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
        c22 = 0.0
        C = np.array([[c11,c12],[c21,c22]])
        Cq = np.matmul(C,np.array([[dq1],[dq2]]))

        # Gravitational matrix
        g11 = M2*G*L1*cos(q1) + M1*G*LC1*cos(q1) + M2*G*LC2*cos(q1+q2)
        g21 = M2*G*LC2*cos(q1+q2)
        Gm = np.array([[g11],[g21]])

        # Damping
        D = self.damp_ratio*np.array([[dq1],[dq2]])

        # Control
        B = np.array([[0.0],[1.0]])

        # Control switch
        if (np.absolute(q1-np.pi/2.) < 0.5):
            # LQR
            # Compute linearized gravity matrix
            dtdq_11 = -G*L1*M2*sin(np.pi/2) - G*LC1*M1*sin(np.pi/2) - G*LC2*M2*sin(np.pi/2)
            dtdq_12 = -G*LC2*M2*sin(np.pi/2)
            dtdq_21 = -G*LC2*M2*sin(np.pi/2)
            dtdq_22 = -G*LC2*M2*sin(np.pi/2)
            dtau_dq = np.array([[dtdq_11,dtdq_12],[dtdq_21,dtdq_22]])

            # Compute linearlized model
            A_lin = np.zeros((4,4),dtype=np.float)
            A_lin[:2,2:] = np.eye(2,dtype=np.float)
            A_lin[2:,:2] = np.matmul(-Minv,dtau_dq)
            A_lin[2:,2:] = np.zeros((2,2),dtype=np.float)
            # A_lin[2:,2:] = np.matmul(-Minv,C)

            MB = np.matmul(Minv,B)
            B_lin = np.zeros((4,1),dtype=np.float)
            B_lin[2,0] = MB[0]
            B_lin[3,0] = MB[1]

            # Compute linearized gains
            try:
                K,S,E = control.lqr(A_lin,B_lin,self.Q,self.R)
                u = -K*np.array([[q1-np.pi/2.],[q2],[dq1],[dq2]])
            except:
                u = 0
        else:
            # Swing up
            (Udes,Kdes) = self.compute_energy([np.pi/2., 0.0, 0., 0.0])
            (Ucur,Kcur) = self.current_energy
            Eerr = (Kcur+Ucur) - (Kdes+Udes)
            u = 4.*dq1*Eerr

        if np.absolute(u) > 30:
            u = np.sign(u)*30

        self.last_u = u
        Bu = B*u

        # Compute dynamics
        xdot = np.matmul(Minv,Bu-Cq-Gm-D)

        dydx[1] = xdot[0]
        dydx[3] = xdot[1]

        return dydx

    def step(self, dt):
        self.state = integrate.odeint(self.dstate_dt, self.state, [0, dt])[1]
        self.update_task_space()
        self.update_energy()
        self.time_elapsed += dt

#------------------------------------------------------------
# set up initial state and global variables
pendulum = Acrobot([-np.pi/2.+0.05, 0.0, 0., 0.0])
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
control_text = ax.text(0.02,0.85, '', transform=ax.transAxes)

def init():
    """initialize animation"""
    line1.set_data([], [])
    line2.set_data([], [])
    time_text.set_text('')
    energy_text.set_text('')
    control_text.set_text('')
    return line1, line2, time_text, energy_text, control_text

def animate(i):
    """perform animation step"""
    global pendulum, dt
    pendulum.step(dt)
    x,y = pendulum.current_position

    line1.set_data([pendulum.origin[0],x[1]],[pendulum.origin[1],y[1]])
    line2.set_data([x[1],x[2]],[y[1],y[2]])

    time_text.set_text('time = %.1f' % pendulum.time_elapsed)
    energy_text.set_text('energy = (U%.3f,K%.3f) J' % pendulum.current_energy)
    control_text.set_text('control = %.3f Nm' % pendulum.last_u)
    return line1, line2, time_text, energy_text, control_text

# choose the interval based on dt and the time to animate one step
from time import time
t0 = time()
animate(0)
t1 = time()
interval = 1000 * dt - (t1 - t0)

ani = animation.FuncAnimation(fig, animate, frames=300,
                              interval=interval, blit=True, init_func=init)
plt.show()
