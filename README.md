[attempt 2.py](https://github.com/user-attachments/files/23287324/attempt.2.py)
'''set up and import libraries necessary to plot'''

import matplotlib.pyplot as plt
import numpy as np

'''
GLOBAL VARIABLES
'''
'''A'''
    t_0 = 4.1 #(s) seconds
    t_a = 0.6 #(s) seconds
    xmin = -20  # Femtometers (fm)
    xmax = 20   # Femtometers (fm)
    
    plot_points = 1000
    steps = 1000
    B = np.arange(0, steps, 1)
    b = np.arange(0, (-1*steps), -1)

'''B'''
    t_min = 0
    t_max = 20
    t = np.linspace(t_min, t_max, steps)
    dt = (t_max - t_min) / steps

'''Frequency domain'''
    f_0 = 10 # Adjust this based on what frequencies you want to see
    f = np.linspace(-f_0, f_0, steps)
    df = (2 * f_0) / steps


'''=Part A
- Define function
- Define graph boundaries
- Call function
- Title graph
- Return Graph
'''

    def woods_saxon(t):
        """
        Calculate Woods-Saxon potential at position(s) x.
        Formula: p(t) = 1/(1+e^{(t-t_0)/t_a})
        """
        return 1 / (1 + np.exp((t - t_0)/ t_a))
    def p(t):
        return woods_saxon(t)
    def graph_woods_saxon():
        """Plot the full Woods-Saxon potential."""
        x_axis = np.linspace(xmin, xmax, plot_points)
        y_axis = woods_saxon(x_axis)
        plt.figure()
        plt.plot(x_axis, y_axis, 'b-', linewidth=2)
        plt.xlabel("Distance (fm)")
        plt.ylabel("Energy (MeV)")
        plt.title("Full Woods-Saxon Potential")
        plt.grid(True, alpha=0.3)
        plt.show()

'''
Part B
~p(f) = int(-infty -> infty)[p(t)e^{i2(pi)ft}dt]
~p(f) = int(-infty -> infty)[(1/(1+e^{(t-t_0)/t_a}))e^{i2(pi)ft}dt]
= sum(-infty -> infty) [(1/(1+e^{(t-t_0)/t_a}))e^{i2(pi)ft}dt]
'''


    def compute_fourier_transform(frequencies):
        """Compute Fourier transform for all frequencies (memory efficient)."""
        # Reshape for broadcasting: t as (steps, 1), frequencies as (1, steps)
        t_col = t.reshape(-1, 1)
        f_row = frequencies.reshape(1, -1)
        
        # Compute integrand with broadcasting
        integrand = p(t_col) * np.exp(1j * 2 * np.pi * f_row * t_col)
        print(np.exp(1j * 2 * np.pi * f_row * t_col))
        # Sum over time axis (axis=0)
        return np.sum(integrand, axis=0) * dt
    
    def power_spectrum(frequencies):
        """Compute power spectrum: |~p(f)|^2"""
        return np.abs(compute_fourier_transform(frequencies))**2
    def norm(frequencies):
        P = power_spectrum(frequencies)
        N = np.sum(P) * df
        print(N)
        return P/N
    


    if __name__ == '__main__':
        ''' Plot Part A: Woods-Saxon potential'''
        graph_woods_saxon()
        
        '''Plot Part B: Fourier transform (vectorized computation)'''
        p_tilde_values = compute_fourier_transform(f)
        
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(18, 5))
        
        '''Real part'''
        ax1.plot(f, np.real(p_tilde_values), linewidth=2)
        ax1.set_xlabel(r"Frequency $f$")
        ax1.set_ylabel(r"Real($\tilde{p}(f)$)")
        ax1.set_title("Fourier Transform (Real Part)")
        ax1.grid(True, alpha=0.3)
        
        ''' Imaginary part'''
        ax2.plot(f, np.imag(p_tilde_values), linewidth=2)
        ax2.set_xlabel(r"Frequency $f$")
        ax2.set_ylabel(r"Imag($\tilde{p}(f)$)")
        ax2.set_title("Fourier Transform (Imaginary Part)")
        ax2.grid(True, alpha=0.3)
        
        '''Power spectrum'''
        ax3.plot(f, power_spectrum(f), linewidth=2)
        ax3.set_xlabel(r"Frequency $f$")
        ax3.set_ylabel(r"$|\tilde{p}(f)|^2$")
        ax3.set_title("Power Spectrum")
        ax3.grid(True, alpha=0.3)
        
        ax4.plot(f, norm(f), linewidth=2)
        ax4.set_xlabel(r"Frequency $f$")
        ax4.set_ylabel(r"Norm $P(f)$")
        ax4.set_title("Normalized Power Spectrum")
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
