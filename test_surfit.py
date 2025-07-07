import numpy as np
from scipy.interpolate import _fitpack

# Simple test data
x = np.array([0, 1, 2, 0, 1, 2])
y = np.array([0, 0, 0, 1, 1, 1])
z = np.array([1, 2, 3, 4, 5, 6])
w = np.ones(6)

xb, xe = 0, 2
yb, ye = 0, 1
kx = ky = 3
task = 0
s = 0
eps = 1e-16

nxest = nyest = 20
tx = np.zeros(nxest)
ty = np.zeros(nyest)

# Calculate workspace
u = nxest - kx - 1
v = nyest - ky - 1
km = max(kx, ky) + 1
ne = max(nxest, nyest)
bx = kx*v + ky + 1
by = ky*u + kx + 1
b1 = bx if bx > by else by
b2 = bx + v - ky if bx > by else by + u - kx
lwrk1 = u*v*(2 + b1 + b2) + 2*(u + v + km*(6 + ne) + ne - kx - ky) + b2 + 1
lwrk2 = u*v*(b2 + 1) + b2
wrk = np.zeros(lwrk1)

print("Calling _surfit...")
result = _fitpack._surfit(x, y, z, w, xb, xe, yb, ye, kx, ky, task, s, eps, tx, ty, nxest, nyest, wrk, lwrk1, lwrk2)
print(f"Result type: {type(result)}")
print(f"Result length: {len(result)}")
print(f"Result: {result}")