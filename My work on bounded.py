import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter


"""
a = (2/l) * ∫ p(x) sin((nπx)/l))

b = (2/nπv) * ∫ q(x) sin((nπx)/l))

p(x) = np.exp(-x**2)

q(x) = 0

------------------------------------------------

I want to solve ∑ (a * cos((nπvt)/l) + b sin((nπvt)/l) * sin((nπx)/l))

which can also be ∑ (a * cos((nπvt)/l) + b sin((nπvt)/l) * ∑ sin((nπx)/l)

where the first sum it the time dependant and the second is position dependant


"""


"""
Varriables
"""
try:
    #equation = (input(f"Input desired initial position as a function of x: "))
    v = int(input(f"Input desired veocity of propigation (m/s): "))
    l = int(input(f"Input desired distance between bounds (m): "))
    
    x_min = 0
    x_max = l

    t_min = 0
    t_max = int(input(f"Input desired time of simulation (s): "))

    n_min = 1
    n_max = int(input(f"Input desired number of nodes: "))
    
    width = int(input(f"Input desired width of Gaussian pulse (m): "))
    height = int(input(f"Input desired amplatude of Gaussian pulse (m): "))

except ValueError:
    print("Invalid input. Using defaults.")
    
    l = 20
    v = 5

    x_min = 0
    x_max = l 

    t_min = 0
    t_max = 30

    n_min = 1
    n_max = 100
    
    width = 5
    height = 5
    #equation = np.exp(-x**2)

fps = 60
t_steps = int(t_max * fps)
interval_ms = 1000 / fps

x_steps = t_steps
print(f"Generating {t_steps} frames for {t_max} seconds of video...")


X = np.linspace(x_min, x_max, x_steps)
T = np.linspace(t_min, t_max, t_steps).reshape(-1, 1)
N = np.arange(1, n_max + 1)

"""
Functions and Computation
"""
print("Calculating physics matrices...")

def p(x):                                                  # Shape: nx1
     
    gaussian = height * np.exp(-width * (x - l/2)**2)
    return gaussian
def q(x):                                                  # Shape: nx1
    return np.zeros_like(x)

def a(n, x):                                               # Shape: nx1
    constant = 2/l
    integrand = p(x) * np.sin((n * np.pi * x) / l)
    integral = np.trapezoid(
        integrand, x)
    return constant * integral

def b(n, x):                                               # Shape: nx1
    constant = 2/(n * np.pi * v)
    integrand = q(x) * np.sin((n * np.pi * x) / l)
    integral = np.trapezoid(integrand, x)
    return constant * integral

def spacial_domain(x,n):                                   # Shape: nx1
    return np.sin((n * np.pi * x) / l)

def time_domain(t,n,x):                                    # Shape: 1xm
    return a(n, x) * np.cos((n * np.pi * v * t) / l) + b(n, x) * np.sin((n * np.pi * v * t) / l)

def summation_over_n(x, t, N):                             # Shape: nxm
    matix_holder = []
    for n in N:
        X, T = np.meshgrid(spacial_domain(x,n), time_domain(t,n,x))
        product = X * T
        matix_holder.append(product)                       # Shape: nxmxlen(n)
    final_matrix = sum(matix_holder)                       # Shape: nxm                 
    return final_matrix

Phi = summation_over_n(X, T, N)



"""
Ploting and Generating an animated graph
"""
print("Setting up animation...")

fig, axis = plt.subplots(figsize=(10, 6))
axis.set_xlim([0, l])
axis.set_ylim([-height*1.2, height*1.2])
axis.set_xlabel("Position (m)")
axis.set_ylabel("Amplitude")
axis.set_title(f"1D Wave Equation (v={v} m/s)")
axis.grid(True, alpha=0.3)

line, = axis.plot([], [], 'b-', linewidth=2)
time_text = axis.text(0.02, 0.95, '', transform=axis.transAxes, fontsize=12,
                      bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))

def init():
    line.set_data([], [])
    time_text.set_text('')
    return line, time_text

def update(frame):
    # Update the wave line
    line.set_data(X, Phi[frame, :])
    
    # Update the clock
    # Use .item() to safely get the float value from the array
    current_time = T[frame].item()
    time_text.set_text(f"Time = {current_time:.2f} s")
    
    return line, time_text

# Create the animation object
anim = FuncAnimation(fig, update, init_func=init, frames=t_steps, blit=True)

# --- 5. SAVE TO GIF (The Rendering Step) ---
print("Rendering GIF... (This might take a minute)")

# writer='pillow' is built-in to matplotlib. 
# fps=60 ensures the file plays back at the speed we calculated for.
anim.save('wave_simulation.gif', writer='pillow', fps=fps)

print("Done! Saved as 'wave_simulation.gif'")