import numericalScemes
import composeAndDecompose
import numpy as np
import math as mt
import matplotlib.pyplot as plt
import imageio

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
            initial_state_one[n] = 1
        elif (x_n <= 8) and (x_n >= 6):
            initial_state_one[n] = (1 - mt.cos(mt.pi*x_n))/2

    initial_state_two = np.ones(x_axis.shape)
    return [initial_state_two, initial_state_one]

def ExersiseOne():
    n = 10
    x_axis = np.linspace(0, n, 50)
    t_axis = np.linspace(0, n, 60)
    initial_state = SampleInitialStateOne(x_axis)

    test1 = numericalScemes.NumericalScheme(initial_state, x_axis, t_axis, -0.5)
    test2 = numericalScemes.NumericalScheme(initial_state, x_axis, t_axis, -0.5)

    for i in range(50):
        plt.figure()
        plt.step(x_axis, test1.state)
        plt.step(x_axis, test2.state)
        test1.TickFromm()
        test2.TickUpwind()
    plt.show()

    return 0

def MakeFig():
    fig, axs = plt.subplots(2, 2)
    ax1 = axs[0,0]
    ax2 = axs[0,1]
    ax3 = axs[1,0]
    ax4 = axs[1,1]

    ax_list = [ax1, ax2, ax3, ax4]

    ax1.set_title("p(t)")
    ax2.set_title("u(t)")
    ax3.set_title("w1(t)")
    ax4.set_title("w2(t)")

    for ax in ax_list:
        ax.grid()
        ax.set_ylim(-.5, 1.5)
        ax.set_xlim(0, 10)
        ax.label_outer()
    return fig, ax1, ax2, ax3, ax4

def ExerciseTwo():
    K0 = 4
    rho0 = 1


    x_axis = np.linspace(0, 10, 50)
    t_axis = np.linspace(0, 10, 120)

    A = np.array([[0, K0], [1/rho0, 0]])
    [u, p] = SampleInitialStateTwo(x_axis)

    twoD_scheme = composeAndDecompose.ComposeDecompose(A, np.array([u, p]), x_axis, t_axis, scheme="VanLeer")
    twoD_scheme1 = composeAndDecompose.ComposeDecompose(A, np.array([u, p]), x_axis, t_axis, scheme="Fromm")
    twoD_scheme2 = composeAndDecompose.ComposeDecompose(A, np.array([u, p]), x_axis, t_axis, scheme="Upwind")

    my_twoD_schemes = [twoD_scheme, twoD_scheme1, twoD_scheme2]
    filenames = []

    for i in range(100):
        [fig, ax1, ax2, ax3, ax4] = MakeFig()
        t = t_axis[i]
        fig.suptitle("t = " + str(round(t,3)))
        for scheme in my_twoD_schemes:
            ax1.step(x_axis, scheme.x[0])
            ax2.step(x_axis, scheme.x[1])
            ax3.step(x_axis, scheme.w[0])
            ax4.step(x_axis, scheme.w[1])
            scheme.March()

        ax2.legend(["Van Leer", "Fromm", "Upwind"], loc='upper right')
        filename = f'{i}.png'
        filenames.append(filename)
        fig.savefig("figures/"+filename)
        plt.close(fig)

    with imageio.get_writer('mygif.gif', mode='I') as writer:
        for filename in filenames:
            image = imageio.imread("figures/" + filename)
            writer.append_data(image)


    return 0

# ExersiseOne()
ExerciseTwo()
