from fenics import *
import sympy as sym
import numpy as np
import matplotlib.pyplot as plt

# Make analytic expression for G with a tolerance
x, y = sym.symbols('x[0] x[1]')

b = np.array([1,0])
a = np.array([0,0])
L = np.linalg.norm(b-a)

epsilon = 1e-10

tau_dot_amx = a[0]-x

tau = b-a

ra = ((x-a[0])**2 + (y-a[1])**2 + epsilon)**0.5 
rb = ((x-b[0])**2 + (y-b[1])**2 + epsilon)**0.5

G = sym.ln( (rb+L+tau_dot_amx)) - sym.ln((ra+tau_dot_amx))

# convert to fenics
G = Expression(sym.printing.ccode(G).replace('log', 'std::log'), degree=2)


# Now plot this function in the domain to see what it looks like
mesh = UnitSquareMesh(1000, 100)
c = mesh.coordinates()
c[:,0] *= 3
c[:,1] *= 2
c -= 1

V = FunctionSpace(mesh, 'CG', 1)

Gi = interpolate(G, V)
Gi.vector().get_local()
plot(Gi)
plt.savefig('Gcolorplot.png')


# We also plot a comparison of G with epsilon and the expression we computed using LHospital
plus_side_xs = np.linspace(1+10*DOLFIN_EPS, 1.5, 200)
Gi_vals = [Gi(x,0) for x in plus_side_xs]

def lhopG(x):
    return np.log(x/(x-1))
lhop_vals = [lhopG(x) for x in plus_side_xs]

plt.plot(plus_side_xs, Gi_vals, 'b', label='G(x,0,0) with tolerance')
plt.plot(plus_side_xs, lhop_vals, 'r:', label='ln(x/x-1)')
plt.xlabel('x')
plt.legend()
plt.savefig('Comparison.png')




