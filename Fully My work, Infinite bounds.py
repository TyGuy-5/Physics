import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

try:
    x_start = int(input("Smallest position bound (e.g., -10 m): "))
    x_end = int(input("Largest position bound (e.g., 10 m): "))
    t_start = int(input("First time bound (e.g., 0 s): "))
    t_end = int(input("Last time bound (e.g., 10 s): "))
    v = int(input("Initial verticle velocity (e.g., 1 m/s): "))
    steps = int(input("Accuracy (steps) (e.g., 300): "))
except ValueError:
    print("Invalid input. Using defaults.")
    x_start, x_end, t_start, t_end, v, steps = -10, 10, 0, 10, 1, 300


x_init = (x_start, x_end)
t_init = (t_start, t_end)
x = np.linspace(x_init[0], x_init[-1], steps)                     #Dimensions: 1xn
t = np.linspace(t_init[0], t_init[-1], steps).reshape(-1,1)       #Dimensions: nx1

X,T = np.meshgrid(x, t)                                           #Dimensions: nxn, nxn


def P(val):
    return np.exp(-val**2)

def q(val):
    return np.zeros_like(val)


integration_domain = np.linspace(X - (v * T), X + (v * T), steps)
integrand = q(integration_domain)
L = np.trapezoid(y=integrand, x=integration_domain, axis=0)

A = P(X - (v * T))              #Dimensions: nxn
B = P(X + (v * T))              #Dimensions: nxn

displacement_term = 0.5 * (A + B)
if v != 0:
    integral_term = (1 / (2 * v)) * L
else:
    integral_term = 0
Phi = displacement_term + integral_term


'''
Graphing
'''

fig, axis = plt.subplots(figsize=(10, 6))


axis.set_xlim([x_start, x_end])
axis.set_ylim([-0.5, 2]) 

axis.set_xlabel("Position (m)")
axis.set_ylabel("Amplitude")
axis.set_title(f"1D Wave Equation (D'Alembert Solution) at v={v} m/s")
axis.grid(True, alpha=0.3)

line, = axis.plot([], [], 'b-', linewidth=2, label='Wave Function')
axis.legend()

def update_data(frame_index):

    y_data = Phi[frame_index, :]
    
    line.set_data(x, y_data)
    return line, 

animation = FuncAnimation(
    fig=fig, 
    func=update_data, 
    frames=len(t), 
    interval=30, 
    repeat=True
)

plt.show()