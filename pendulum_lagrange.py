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

# Input torque
tau = Symbol('tau')

# Initialize dimensional parameters
m1 = Symbol('m1')
l1 = Symbol('l1')
I1 = Symbol('I1')
lc1 = Symbol('lc1')

# Physical constants
g = Symbol('g')
t = Symbol('t')

# Define rotation matrices
R01 = Matrix([[cos(q1), -sin(q1), 0.0],[sin(q1),cos(q1),0.0],[0.0,0.0,1.0]])

# Define local angular velocities
omega0_01 = Matrix([[0.0],[0.0],[dq1]])

# Links CoM position in local coordinates
r1c1 = Matrix([[lc1],[0.0],[0.0]])

# Rotate CoM positions into global coordinates
r0c1 = R01*r1c1

# Define location of end of the first link in local coordinates
r1p = Matrix([[l1],[0.0],[0.0]])

# Compute velocity of CoM in global coordinates
v0c1 = cross_skew(omega0_01)*r0c1

# Define kinetic energy of each link
K1 = 0.5*m1*vsq(v0c1) + 0.5*I1*vsq(omega0_01)

# Define the potential energy of each link
P1 = m1*g*lc1*sin(q1)

# Define the Lagrangian
L = K1[0] - P1

# Solve equations of motion
eq_motion = simplify(Eq(diff(diff(L,dq1),t)-diff(L,q1),tau))
pprint(eq_motion)
