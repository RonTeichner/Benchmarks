import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

def P_x0_t0_arg(theta_x0_t0):
    th = theta_x0_t0
    if th == 0:
        res = np.multiply(1-th, np.log(1-th))
    elif th == 1:
        res = np.multiply(th, np.log(th))
    else:
        res = np.multiply(th, np.log(th)) + np.multiply(1-th, np.log(1-th))
    return res

def P_x0_t1_arg(theta_x0_t1):
    return P_x0_t0_arg(theta_x0_t1)

def P_t0_arg(theta_x0_t0, theta_x0_t1, p_t0):
    p_t1 = 1 - p_t0
    th0, th1 = theta_x0_t0, theta_x0_t1
    e_th = p_t0*th0 + p_t1*th1
    return np.multiply(th0, np.log(e_th)) + np.multiply(1 - th0, np.log(1 - e_th))

def P_t1_arg(theta_x0_t0, theta_x0_t1, p_t0):
    p_t1 = 1 - p_t0
    th0, th1 = theta_x0_t0, theta_x0_t1
    e_th = p_t0 * th0 + p_t1 * th1
    return np.multiply(th1, np.log(e_th)) + np.multiply(1 - th1, np.log(1 - e_th))

def mutualInfoSpecific_x(p_t0, p_x0_t0, theta_x0_t0, theta_x0_t1):
    p_t1 = 1 - p_t0
    p_x0_t1 = 1 - p_x0_t0
    return p_x0_t0*P_x0_t0_arg(theta_x0_t0) + p_x0_t1*P_x0_t1_arg(theta_x0_t1) - (p_t0*P_t0_arg(theta_x0_t0, theta_x0_t1, p_t0) + p_t1*P_t1_arg(theta_x0_t0, theta_x0_t1, p_t0))

def Delta_mutualInfoSpecific_x(p_t0, p_x0_t0, theta0_x0_t0, theta0_x0_t1, theta1_x0_t0, theta1_x0_t1):
    return mutualInfoSpecific_x(p_t0, p_x0_t0, theta1_x0_t0, theta1_x0_t1) - mutualInfoSpecific_x(p_t0, p_x0_t0, theta0_x0_t0, theta0_x0_t1)

def LogLikelihood(p_x0_t0, theta_x0_t0, theta_x0_t1, p_y_x0_0):
    p_x0_t1 = 1 - p_x0_t0
    return (np.multiply(p_x0_t0, p_y_x0_0*np.log(theta_x0_t0) + (1-p_y_x0_0)*np.log(1-theta_x0_t0)) + np.multiply(p_x0_t1, np.log(theta_x0_t1)))

def Delta_L0(p_x0_t0, theta0_x0_t0, theta0_x0_t1, theta1_x0_t0, theta1_x0_t1, p_y_x0_0):
    return LogLikelihood(p_x0_t0, theta1_x0_t0, theta1_x0_t1, p_y_x0_0) - LogLikelihood(p_x0_t0, theta0_x0_t0, theta0_x0_t1, p_y_x0_0)


p_t0 = 0.5
p_y_x0_0 = 0.3
theta0_x0_t1, theta1_x0_t1 = 1, 1
p_t1 = 1 - p_t0

res = 1e-2
p_x0_t0 = np.arange(0, 1, res)

# the best value for theta_x0_t0 is 0.
# let's assume that the model has made a small update 0f 1e-2 (in the correct direction)

modelUpdate = -1e-2 # update in the correct direction
theta0_x0_t0 = np.arange(np.abs(modelUpdate), 1, res)
theta1_x0_t0 = theta0_x0_t0 + modelUpdate

delta_L0, delta_mI, mI, L0 = np.zeros((theta0_x0_t0.shape[0], p_x0_t0.shape[0])), np.zeros((theta0_x0_t0.shape[0], p_x0_t0.shape[0])), np.zeros((theta0_x0_t0.shape[0], p_x0_t0.shape[0])), np.zeros((theta0_x0_t0.shape[0], p_x0_t0.shape[0]))
for p_x0_t0_idx, p_x0_t0_val in enumerate(p_x0_t0):
    for theta0_x0_t0_idx, theta0_x0_t0_val in enumerate(theta0_x0_t0):
        delta_L0[theta0_x0_t0_idx, p_x0_t0_idx] = Delta_L0(p_x0_t0_val, theta0_x0_t0_val, theta0_x0_t1, theta1_x0_t0[theta0_x0_t0_idx], theta1_x0_t1, p_y_x0_0)
        delta_mI[theta0_x0_t0_idx, p_x0_t0_idx] = Delta_mutualInfoSpecific_x(p_t0, p_x0_t0_val, theta0_x0_t0_val, theta0_x0_t1, theta1_x0_t0[theta0_x0_t0_idx], theta1_x0_t1)
        mI[theta0_x0_t0_idx, p_x0_t0_idx] = mutualInfoSpecific_x(p_t0, p_x0_t0_val, theta0_x0_t0_val, theta0_x0_t1)
        L0[theta0_x0_t0_idx, p_x0_t0_idx] = LogLikelihood(p_x0_t0_val, theta0_x0_t0_val, theta0_x0_t1, p_y_x0_0)

fig = plt.figure()
ax = fig.gca(projection='3d')

# Make data.
X = p_x0_t0  # np.arange(-5, 5, 0.25)
Y = theta0_x0_t0  # np.arange(-5, 5, 0.25)
X, Y = np.meshgrid(X, Y)
# R = np.sqrt(X**2 + Y**2)
# Z = np.sin(R)

# Plot the surface.
surf = ax.plot_surface(X, Y, delta_L0, cmap=cm.coolwarm, linewidth=0, antialiased=False)

# Customize the z axis.
# ax.set_zlim(-1.01, 1.01)
# ax.zaxis.set_major_locator(LinearLocator(10))
# ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# Add a color bar which maps values to colors.
ax.set_xlabel('p(t=0 | x_0)')
ax.set_ylabel('p(y_hat=1 | x_0, t=0)')
ax.set_title('Delta LogLikelihood')
fig.colorbar(surf, shrink=0.5, aspect=5)
#plt.show()

fig = plt.figure()
ax = fig.gca(projection='3d')
# Plot the surface.
surf = ax.plot_surface(X, Y, delta_mI, cmap=cm.coolwarm, linewidth=0, antialiased=False)
# Add a color bar which maps values to colors.
ax.set_xlabel('p(t=0 | x_0)')
ax.set_ylabel('p(y_hat=1 | x_0, t=0)')
ax.set_title('Delta Mutual-Information')
fig.colorbar(surf, shrink=0.5, aspect=5)

# since p_y_x0_0 = 0.3 and the change in the model is negative (decreasing the probability of y=1 given t=0), we expect that if the probability before the change was higher than
# 0.3 the change will increase the logLikelihood and vice versa. The next figure shows exactly that:
fig = plt.figure()
plt.subplot(2, 2, 1)
gridVals = np.arange(0.1, 1, 0.2)
for gridVal in gridVals:
    p_x0_t0_05_idx = np.argmin(np.abs(p_x0_t0 - gridVal))
    plt.plot(theta0_x0_t0, delta_L0[:, p_x0_t0_05_idx], label='p(t=0 | x_0) = %0.2f' % gridVal)
plt.xlabel('p(y_hat=1 | x_0, t=0)')
plt.title('Delta LogLikelihood')
plt.grid(True)
plt.legend()

plt.subplot(2, 2, 2)
for gridVal in gridVals:
    p_x0_t0_05_idx = np.argmin(np.abs(p_x0_t0 - gridVal))
    plt.plot(theta0_x0_t0, delta_mI[:, p_x0_t0_05_idx], label='p(t=0 | x_0) = %0.2f' % gridVal)
plt.xlabel('p(y_hat=1 | x_0, t=0)')
plt.title('Delta Mutual-Information')
plt.grid(True)
plt.legend()

plt.subplot(2, 2, 4)
for gridVal in gridVals:
    p_x0_t0_05_idx = np.argmin(np.abs(p_x0_t0 - gridVal))
    plt.plot(theta0_x0_t0, mI[:, p_x0_t0_05_idx], label='p(t=0 | x_0) = %0.2f' % gridVal)
plt.xlabel('p(y_hat=1 | x_0, t=0)')
plt.title('Mutual-Information')
plt.grid(True)
plt.legend()

plt.subplot(2, 2, 3)
gridVals = np.arange(0.1, 1, 0.2)
for gridVal in gridVals:
    p_x0_t0_05_idx = np.argmin(np.abs(p_x0_t0 - gridVal))
    plt.plot(theta0_x0_t0, L0[:, p_x0_t0_05_idx], label='p(t=0 | x_0) = %0.2f' % gridVal)
plt.xlabel('p(y_hat=1 | x_0, t=0)')
plt.title('LogLikelihood')
plt.grid(True)
plt.legend()

plt.show()
