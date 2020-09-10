import time
import numpy as np
from sympy import *
from sympy.physics.mechanics import *

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
b = Symbol('b')

# Variable matrices
qdot = Matrix([[dq1],[dq2]])

# Mass matrix
m11 = I1 + I2 + m2*l1**2 + 2.0*m2*l1*lc2*cos(q2) + m1*lc1**2 + m2*lc2**2
m12 = I2 + m2*l1*lc2*cos(q2) + m2*lc2**2
m21 = I2 + m2*l1*lc2*cos(q2) + m2*lc2**2
m22 = I2 + m2*lc2**2
M = Matrix([[m11,m12],[m21,m22]])
Minv = M**-1

# Corriolis matrix
c11 = -2*m2*l1*lc2*sin(q2)*dq2
c12 = -m2*l1*lc2*sin(q2)*dq2
c21 = m2*l1*lc2*sin(q2)*dq1
c22 = 0
C = Matrix([[c11,c12],[c21,c22]])
Cq = C*qdot

# Gravitational matrix
g11 = m2*g*l1*cos(q1) + m1*g*lc1*cos(q1) + m2*g*lc2*cos(q1+q2)
g21 = m2*g*lc2*cos(q1+q2)
G = Matrix([[g11],[g21]])

# Damping
Dq = b*qdot

# Input
B = Matrix([[0.],[1.]])
Bu = B*tau

# Compute linearized model
dGdq_11 = diff(g11,q1)
dGdq_12 = diff(g11,q2)
dGdq_21 = diff(g21,q1)
dGdq_22 = diff(g21,q2)
dGdq = Matrix([[diff(g11,q1),dGdq_12],[dGdq_21,dGdq_22]])

dBdq_11 = 0.
dBdq_12 = 0.
dBdq_21 = diff(Bu[1],q1)
dBdq_22 = diff(Bu[1],q2)
dBdq = Matrix([[dBdq_11,dBdq_12],[dBdq_21,dBdq_22]])

A11 = Matrix([[0.,0.],[0.,0.]])
A12 = Matrix([[1.,0.],[0.,1.]])
A21 = dGdq + dBdq
A22 = Matrix([[0.,0.],[0.,0.]])
A_lin = Matrix([[A11,A12],[A21,A22]])
B_lin = Matrix([[0.],[0.],B])

pprint(A_lin)
print('=======================================================================')
pprint(B_lin)
