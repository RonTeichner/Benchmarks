import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from mutualInfoTrain_func import *
from matplotlib.ticker import LinearLocator, FormatStrFormatter

def mutualInfoSpecific_x(p_x0_t0, theta_x0_t0, theta_x0_t1):
    p_x0_t1 = 1 - p_x0_t0
    alpha = p_x0_t1*theta_x0_t1 + p_x0_t0*theta_x0_t0
    if theta_x0_t1 == 1:
        P_x0_t1_arg = theta_x0_t1*(np.log(theta_x0_t1) - np.log(alpha))
    else:
        P_x0_t1_arg = theta_x0_t1*(np.log(theta_x0_t1) - np.log(alpha)) + (1-theta_x0_t1)*(np.log(1-theta_x0_t1) - np.log(1-alpha))

    if theta_x0_t0 == 0:
        P_x0_t0_arg = (1-theta_x0_t0)*(np.log(1-theta_x0_t0) - np.log(1-alpha))
    else:
        P_x0_t0_arg = theta_x0_t0*(np.log(theta_x0_t0) - np.log(alpha)) + (1-theta_x0_t0)*(np.log(1-theta_x0_t0) - np.log(1-alpha))
    return p_x0_t1*P_x0_t1_arg + p_x0_t0*P_x0_t0_arg

def Delta_mutualInfoSpecific_x(p_x0_t0, theta0_x0_t0, theta0_x0_t1, theta1_x0_t0, theta1_x0_t1):
    return mutualInfoSpecific_x(p_x0_t0, theta1_x0_t0, theta1_x0_t1) - mutualInfoSpecific_x(p_x0_t0, theta0_x0_t0, theta0_x0_t1)

def LogLikelihood(p_x0_t0, theta_x0_t0, theta_x0_t1, p_y_x0_0):
    p_x0_t1 = 1 - p_x0_t0
    return (np.multiply(p_x0_t0, p_y_x0_0*np.log(theta_x0_t0) + (1-p_y_x0_0)*np.log(1-theta_x0_t0)) + np.multiply(p_x0_t1, np.log(theta_x0_t1)))

def Delta_L0(p_x0_t0, theta0_x0_t0, theta0_x0_t1, theta1_x0_t0, theta1_x0_t1, p_y_x0_0):
    return LogLikelihood(p_x0_t0, theta1_x0_t0, theta1_x0_t1, p_y_x0_0) - LogLikelihood(p_x0_t0, theta0_x0_t0, theta0_x0_t1, p_y_x0_0)

def mutualInfoSpecific_x_lossF(p_x0_t0, theta_x0_t0, theta_x0_t1):
    p_x0_t1 = 1 - p_x0_t0
    _, mutualInfo, _ = loss_function_PS(torch.zeros(1), torch.tensor(theta_x0_t1, dtype=torch.float), torch.tensor(theta_x0_t0, dtype=torch.float), torch.zeros(1), torch.tensor(p_x0_t1, dtype=torch.float), True)
    return mutualInfo.detach().numpy()

p_t0 = 0.5
p_y_x0_0 = 0.3
theta0_x0_t1, theta1_x0_t1 = 0.99, 0.99
p_t1 = 1 - p_t0

res = 1e-2
p_x0_t0 = np.arange(res, 1, res)

# the best value for theta_x0_t0 is 0.
# let's assume that the model has made a small update 0f 1e-2 (in the correct direction)

modelUpdate = -1e-2 # update in the correct direction
theta0_x0_t0 = np.arange(np.abs(modelUpdate), 1, res)
theta1_x0_t0 = theta0_x0_t0 + modelUpdate

delta_L0, delta_mI, mI, mI_lossFunc, L0 = np.zeros((theta0_x0_t0.shape[0], p_x0_t0.shape[0])), np.zeros((theta0_x0_t0.shape[0], p_x0_t0.shape[0])), np.zeros((theta0_x0_t0.shape[0], p_x0_t0.shape[0])), np.zeros((theta0_x0_t0.shape[0], p_x0_t0.shape[0])), np.zeros((theta0_x0_t0.shape[0], p_x0_t0.shape[0]))
for p_x0_t0_idx, p_x0_t0_val in enumerate(p_x0_t0):
    for theta0_x0_t0_idx, theta0_x0_t0_val in enumerate(theta0_x0_t0):
        delta_L0[theta0_x0_t0_idx, p_x0_t0_idx] = Delta_L0(p_x0_t0_val, theta0_x0_t0_val, theta0_x0_t1, theta1_x0_t0[theta0_x0_t0_idx], theta1_x0_t1, p_y_x0_0)
        delta_mI[theta0_x0_t0_idx, p_x0_t0_idx] = Delta_mutualInfoSpecific_x(p_x0_t0_val, theta0_x0_t0_val, theta0_x0_t1, theta1_x0_t0[theta0_x0_t0_idx], theta1_x0_t1)
        mI[theta0_x0_t0_idx, p_x0_t0_idx] = mutualInfoSpecific_x(p_x0_t0_val, theta0_x0_t0_val, theta0_x0_t1)
        mI_lossFunc[theta0_x0_t0_idx, p_x0_t0_idx] = mutualInfoSpecific_x_lossF(p_x0_t0_val, theta0_x0_t0_val, theta0_x0_t1)
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
ax.set_xlabel('1-e(x0)')
ax.set_ylabel('p(y_hat=1 | x_0, t=0)')
ax.set_title('Delta LogLikelihood')
fig.colorbar(surf, shrink=0.5, aspect=5)
#plt.show()

fig = plt.figure()
ax = fig.gca(projection='3d')
# Plot the surface.
surf = ax.plot_surface(X, Y, delta_mI, cmap=cm.coolwarm, linewidth=0, antialiased=False)
# Add a color bar which maps values to colors.
ax.set_xlabel('1-e(x0)')
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
    plt.plot(theta0_x0_t0, delta_L0[:, p_x0_t0_05_idx], label='e(x0) = %0.2f' % (1-gridVal))
plt.xlabel('p(y_hat=1 | x_0, t=0)')
plt.title('Delta LogLikelihood')
plt.ylim(-0.1, 0.1)
plt.grid(True)
plt.legend()

plt.subplot(2, 2, 2)
for gridVal in gridVals:
    p_x0_t0_05_idx = np.argmin(np.abs(p_x0_t0 - gridVal))
    plt.plot(theta0_x0_t0, delta_mI[:, p_x0_t0_05_idx], label='e(x0) = %0.2f' % (1-gridVal))
plt.xlabel('p(y_hat=1 | x_0, t=0)')
plt.title('Delta Mutual-Information')
plt.grid(True)
plt.legend()

plt.subplot(2, 2, 4)
for gridVal in gridVals:
    p_x0_t0_05_idx = np.argmin(np.abs(p_x0_t0 - gridVal))
    plt.plot(theta0_x0_t0, mI[:, p_x0_t0_05_idx], label='e(x0) = %0.2f' % (1-gridVal))
plt.xlabel('p(y_hat=1 | x_0, t=0)')
plt.title('Mutual-Information')
plt.grid(True)
plt.legend()

plt.subplot(2, 2, 3)
gridVals = np.arange(0.1, 1, 0.2)
for gridVal in gridVals:
    p_x0_t0_05_idx = np.argmin(np.abs(p_x0_t0 - gridVal))
    plt.plot(theta0_x0_t0, L0[:, p_x0_t0_05_idx], label='e(x0) = %0.2f' % (1-gridVal))
plt.xlabel('p(y_hat=1 | x_0, t=0)')
plt.title('LogLikelihood')
plt.grid(True)
plt.legend()

fig = plt.figure()
for gridVal in gridVals:
    p_x0_t0_05_idx = np.argmin(np.abs(p_x0_t0 - gridVal))
    plt.plot(theta0_x0_t0, np.abs(delta_mI[:, p_x0_t0_05_idx]/delta_L0[:, p_x0_t0_05_idx]), label='e(x0) = %0.2f' % (1-gridVal))
plt.xlabel('p(y_hat=1 | x_0, t=0)')
plt.title('delta_mI / delta_L0')
plt.grid(True)
plt.ylim(0, 2)
plt.legend()

plt.show()
