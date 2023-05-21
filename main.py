import numericalScemes
import composeAndDecompose
import numpy as np
import math as mt
import matplotlib.pyplot as plt

def SampleInitialStateOne(x_axis):
    initial_state_one = np.zeros(x_axis.shape)
    for n, x_n in enumerate(x_axis):
        if (x_n <= 4) and (x_n >= 2):
            initial_state_one[n] = 1
        elif (x_n <= 8) and (x_n >= 6):
            initial_state_one[n] = (1 - mt.cos(mt.pi*x_n))/2
    return initial_state_one

def SampleInitialStateTwo(x_axis):
    initial_state_one = np.zeros(x_axis.shape)
    for n, x_n in enumerate(x_axis):
        if (x_n <= 4) and (x_n >= 2):
            initial_state_one[n] = 0
        elif (x_n <= 8) and (x_n >= 6):
            initial_state_one[n] = (1 - mt.cos(mt.pi*x_n))/2

    initial_state_two = np.ones(x_axis.shape)
    return [initial_state_two, initial_state_one]

def ExcersiseOne():
    n = 10
    x_axis = np.linspace(0, n, 30)
    t_axis = np.linspace(0, n, 20)
    initial_state = SampleInitialStateOne(x_axis)

    test1 = numericalScemes.NumericalScheme(initial_state, x_axis, t_axis, 0.5)
    test2 = numericalScemes.NumericalScheme(initial_state, x_axis, t_axis, 0.5)

    for i in range(5):
        plt.figure()
        plt.step(x_axis, test1.state)
        plt.step(x_axis, test2.state)
        plt.show()
        test1.TickFromm()
        test2.TickUpwind()

    return 0

def ExerciseTwo():
    K0 = 1
    rho0 = 1


    x_axis = np.linspace(0, 10, 100)
    t_axis = np.linspace(0, 10, 500)

    A = np.array([[0, K0], [1/rho0, 0]])
    [u, p] = SampleInitialStateTwo(x_axis)

    twoD_scheme = composeAndDecompose.ComposeDecompose(A, np.array([u, p]), x_axis, t_axis)

    for i in range(10):
        fig, (ax1, ax2) = plt.subplots(1, 2)
        for state in twoD_scheme.x:
            ax1.plot(x_axis, state)
        for decomposed_state in twoD_scheme.w:
            ax2.plot(x_axis, decomposed_state)
        plt.show()
        twoD_scheme.March()

    return 0

ExerciseTwo()
