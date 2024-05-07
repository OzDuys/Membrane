import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft, ifft
from matplotlib import cm

N = 40
M = 100
dte = 2 * np.pi / M
t = dte * np.arange(M+1)
h = 1 / N
ta = h ** 2
r = h * np.arange(N+1)
rr, tt = np.meshgrid(r, t)
xx, yy = rr * np.cos(tt), rr * np.sin(tt)

# Define the initial condition
u0 = 1/np.cosh(rr)**2
#u0 = 1 / (rr**2 + 1) * np.cos(5 * tt)
#u0 = -rr**2 * np.sin(tt / 2) + np.sin(6 * tt) * np.cos(tt / 2)**2

u1 = u0.copy()
hta2 = ta**2 / h**2
nn = np.concatenate((np.arange(0, M//2+1), np.arange(-M//2, 0)))**2  # Corrected to include M+1 elements

# Set up the figure and axes
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x_limit = [np.min(xx), np.max(xx)]
y_limit = [np.min(yy), np.max(yy)]
z_limit = [-2, 4]  # Adjust based on your data

for time in range(1, 5001):
    u1t = np.zeros_like(u1)
    for col in range(N+1):
        u1t[:, col] = np.real(ifft(nn * fft(u1[:, col])))
        u1t[0, col] = u1t[-1, col]
    u2 = np.zeros_like(u1)
    # Ensure the slices are of the same length by adjusting the indices
    u2[:, 1:N] = -u0[:, 1:N] + 2*u1[:, 1:N] \
        + hta2 * (u1[:, :N-1] - 2*u1[:, 1:N] + u1[:, 2:N+1]) \
        + hta2/2 * (u1[:, 2:N+1] - u1[:, :N-1]) / np.arange(1, N) \
        - hta2 * u1t[:, 1:N] / np.arange(1, N)**2
    u2[:, N] = np.zeros(M+1)
    u2[:, 0] = np.mean(u2[:, 1]) * np.ones(M+1)

    u0 = u1.copy()
    u1 = u2.copy()

    if time % 10 == 0:
        ax.clear()
        ax.plot_surface(xx, yy, u2, cmap=cm.jet, shade=True)
        ax.set_xlim(x_limit)
        ax.set_ylim(y_limit)
        ax.set_zlim(z_limit)
        ax.view_init(30, 60)
        plt.draw()
        plt.pause(0.001)

plt.show()


