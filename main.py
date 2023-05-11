import numericalScemes
import numpy as np
import math as mt
import matplotlib.pyplot as plt

def SampleInitialState(x_axis):
    initial_state = np.zeros(x_axis.shape)
    for n, x_n in enumerate(x_axis):
        if (x_n <= 4) and (x_n >= 2):
            initial_state[n] = 1
        elif (x_n <= 8) and (x_n >= 6):
            initial_state[n] = (1 - mt.cos(mt.pi*x_n))/2
    return initial_state

n = 10
x0 = np.arange(n)
x_axis = np.linspace(0, 10, 30)
t_axis = np.arange(3)
initial_state = SampleInitialState(x_axis)

test1 = numericalScemes.NumericalSimulation(initial_state, x_axis, t_axis, 0.5)
for i in range(5):
    plt.figure()
    plt.step(x_axis, test1.state)
    plt.show()
    test1.Tick()



