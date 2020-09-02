import time
import numpy as np
from sympy import *
from sympy.physics.mechanics import *

def cross_skew(q):
    return Matrix([[0.0,-q[2],q[1]],[q[2],0.0,-q[0]],[-q[1],q[0],0.0]])

def vsq(v):
    return v.transpose()*v

# Initialize generalized coordinates
q1 = dynamicsymbols('q1')
dq1 = dynamicsymbols('q1',1)
q2 = dynamicsymbols('q2')
dq2 = dynamicsymbols('q2',1)

# Generalized inputs
tau = Symbol('tau')

# Initialize dimensional parameters
m1 = Symbol('m1')
m2 = Symbol('m2')
l1 = Symbol('l1')
l2 = Symbol('l2')
I1 = Symbol('I1')
I2 = Symbol('I2')
lc1 = Symbol('lc1')
lc2 = Symbol('lc2')

# Physical constants
g = Symbol('g')
t = Symbol('t')

# Define world frame and origin
N = ReferenceFrame('N')
pN = Point('N*')
pN.set_vel(N,0)

# Define first-link aligned coordinate frame
A = N.orientnew('A','axis',[q1,N.z])

# Define rotation matrices
R01 = Matrix([[cos(q1), -sin(q1), 0.0],[sin(q1),cos(q1),0.0],[0.0,0.0,1.0]])
R12 = Matrix([[cos(q2), -sin(q2), 0.0],[sin(q2),cos(q2),0.0],[0.0,0.0,1.0]])

# Define local angular velocities
omega0_01 = Matrix([[0.0],[0.0],[dq1]])
omega1_12 = Matrix([[0.0],[0.0],[dq2]])

# Compute link2 angular velocity in gloabl frame
omega0_02 = omega0_01 + R01*omega1_12

# Links CoM position in local coordinates
r1c1 = Matrix([[lc1],[0.0],[0.0]])
r2c2 = Matrix([[lc2],[0.0],[0.0]])

# Rotate CoM positions into global coordinates
r0c1 = R01*r1c1
r0c2 = R01*R12*r2c2

# Define location of end of the first link in local coordinates
r1p = Matrix([[l1],[0.0],[0.0]])
# Rotate end of first link into global coordinate frame
r0p = R01*r1p

# Compute velocity of CoM in global coordinates
v0c1 = cross_skew(omega0_01)*r0c1
v0c2 = cross_skew(omega0_01)*r0p + cross_skew(omega0_02)*r0c2

# Define kinetic energy of each link
K1 = 0.5*m1*vsq(v0c1) + 0.5*I1*vsq(omega0_01)
K2 = 0.5*m2*vsq(v0c2) + 0.5*I2*vsq(omega0_02)

# Define the potential energy of each link
P1 = m1*g*lc1*sin(q1)
P2 = m2*g*(l1*sin(q1) + lc2*sin(q1+q2))

# Define the Lagrangian
L = (K1[0]+K2[0]) - (P1+P2)

# Solve equations of motion
eq_motion1 = simplify(Eq(diff(diff(L,dq1),t)-diff(L,q1),0))
eq_motion1 = simplify(Eq(diff(diff(L,dq1),t)-diff(L,q1),tau))

pprint(eq_motion1)
