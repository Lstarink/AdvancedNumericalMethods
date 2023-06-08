import numericalScemes
import composeAndDecompose
import numpy as np
import math as mt
import matplotlib.pyplot as plt
import imageio
import tqdm
from matplotlib.transforms import offset_copy


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

def ExerciseTwoGif():
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

    for i in tqdm(range(100)):
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


def ExerciseFour1D(x_axis, x_axis_analytical, t_axis, CFL):

    dt = t_axis[1]-t_axis[0]
    initial_state = SampleInitialStateOne(x_axis)

    schemeUpwind = numericalScemes.NumericalScheme(initial_state, x_axis, t_axis, 1, scheme="Upwind")
    schemeFromm = numericalScemes.NumericalScheme(initial_state, x_axis, t_axis, 1, scheme="Fromm")
    schemeVanLeer = numericalScemes.NumericalScheme(initial_state, x_axis, t_axis, 1, scheme="VanLeer")


    fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(15, 20))

    for i in range(6):
        x_shifted = x_axis_analytical - dt*i*np.ones(x_axis_analytical.shape)
        analytical_solution = SampleInitialStateOne(x_shifted)
        if i == 1:
            print(x_axis.shape, schemeUpwind.state[0].shape)
            axs[0, 0].scatter(x_axis, schemeUpwind.state[0], marker=".", color="r")
            axs[0, 1].scatter(x_axis, schemeFromm.state[0], marker=".", color="r")
            axs[0, 2].scatter(x_axis, schemeVanLeer.state[0], marker=".", color="r",label="numerical")
            axs[0, 0].plot(x_axis, schemeUpwind.state[0], linestyle="dashed", color="tab:red", linewidth='.8')
            axs[0, 1].plot(x_axis, schemeFromm.state[0], linestyle="dashed", color="tab:red", linewidth='.8')
            axs[0, 2].plot(x_axis, schemeVanLeer.state[0], linestyle="dashed", color="tab:red", linewidth='.8')
            axs[0, 0].plot(x_axis_analytical, analytical_solution, linestyle="dashed")
            axs[0, 1].plot(x_axis_analytical, analytical_solution, linestyle="dashed")
            axs[0, 2].plot(x_axis_analytical, analytical_solution, linestyle="dashed", label="analytical")
            axs[0, 2].legend(loc="upper right", borderpad=.1)
        if i == 2:
            axs[1, 0].scatter(x_axis, schemeUpwind.state, marker=".", color="r")
            axs[1, 1].scatter(x_axis, schemeFromm.state, marker=".", color="r")
            axs[1, 2].scatter(x_axis, schemeVanLeer.state, marker=".", color="r")
            axs[1, 0].plot(x_axis, schemeUpwind.state, linestyle="dashed", color="tab:red", linewidth='.8')
            axs[1, 1].plot(x_axis, schemeFromm.state, linestyle="dashed", color="tab:red", linewidth='.8')
            axs[1, 2].plot(x_axis, schemeVanLeer.state, linestyle="dashed", color="tab:red", linewidth='.8')
            axs[1, 0].plot(x_axis_analytical, analytical_solution, linestyle="dashed")
            axs[1, 1].plot(x_axis_analytical, analytical_solution, linestyle="dashed")
            axs[1, 2].plot(x_axis_analytical, analytical_solution, linestyle="dashed")
        if i == 5:
            axs[2, 0].scatter(x_axis, schemeUpwind.state, marker=".", color="r")
            axs[2, 1].scatter(x_axis, schemeFromm.state, marker=".", color="r")
            axs[2, 2].scatter(x_axis, schemeVanLeer.state, marker=".", color="r")
            axs[2, 0].plot(x_axis, schemeUpwind.state, linestyle="dashed", color="tab:red", linewidth='.8')
            axs[2, 1].plot(x_axis, schemeFromm.state, linestyle="dashed", color="tab:red", linewidth='.8')
            axs[2, 2].plot(x_axis, schemeVanLeer.state, linestyle="dashed", color="tab:red", linewidth='.8')
            axs[2, 0].plot(x_axis_analytical, analytical_solution, linestyle="dashed")
            axs[2, 1].plot(x_axis_analytical, analytical_solution, linestyle="dashed")
            axs[2, 2].plot(x_axis_analytical, analytical_solution, linestyle="dashed")

        schemeUpwind.Tick()
        schemeVanLeer.Tick()
        schemeFromm.Tick()


    for ax1 in axs:
        for ax in ax1:
            ax.grid()
    #         ax.set_ylim(-0.2, 1.2)

    cols = ["Upwind", "Fromm", "van Leer"]
    rows = ["N = 1", "N = 2", "N = 5"]

    plt.setp(axs[2,:], xlabel='X')
    plt.setp(axs[:,0], ylabel='q')

    pad = 5  # in points

    for ax, col in zip(axs[0], cols):
        ax.annotate(col, xy=(0.5, 1), xytext=(0, pad),
                    xycoords='axes fraction', textcoords='offset points',
                    size='large', ha='center', va='baseline')

    for ax, row in zip(axs[:, 0], rows):
        ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
                    xycoords=ax.yaxis.label, textcoords='offset points',
                    size='large', ha='right', va='center')

    fig.tight_layout()
    # tight_layout doesn't take these labels into account. We'll need
    # to make some room. These numbers are are manually tweaked.
    # You could automatically calculate them, but it's a pain.
    fig.subplots_adjust(left=0.15, top=0.95)
    fig.savefig("figuresEx4/CFL" + str(CFL))

    plt.show()

    return 0

def f(x_axis, t):
    x_shifted = x_axis_analytical - t * np.ones(x_axis_analytical.shape)
    sampled = SampleInitialStateOne(x_shifted)
    return sampled

def ExerciseFour2D(x_axis, x_axis_analytical, t_axis, CFL):
    K0 = 4
    rho0 = 1

    A = np.array([[0, K0], [1 / rho0, 0]])
    [u, p] = SampleInitialStateTwo(x_axis)

    schemeUpwind = composeAndDecompose.ComposeDecompose(A, np.array([u, p]), x_axis, t_axis, scheme="VanLeer")
    schemeFromm = composeAndDecompose.ComposeDecompose(A, np.array([u, p]), x_axis, t_axis, scheme="Fromm")
    schemeVanLeer = composeAndDecompose.ComposeDecompose(A, np.array([u, p]), x_axis, t_axis, scheme="Upwind")

    my_twoD_schemes = [schemeUpwind, schemeFromm, schemeVanLeer]
    fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(15, 20))

    for i in range(6):
        analytical_p = 0.5*(f(x_axis_analytical, dt*i*2) + f(x_axis_analytical, -dt*i*2))
        analytical_u = 1 + 0.5*(-f(x_axis_analytical, dt*i*2) + f(x_axis_analytical, -dt*i*2))
        if i == 1:
            axs[0, 0].scatter(x_axis, schemeUpwind.x[0], marker=".", color="r")
            axs[0, 1].scatter(x_axis, schemeFromm.x[0], marker=".", color="r")
            axs[0, 2].scatter(x_axis, schemeVanLeer.x[0], marker=".", color="r",label="numerical u")
            axs[0, 0].plot(x_axis, schemeUpwind.x[0], linestyle="dashed", color="tab:red", linewidth='.8')
            axs[0, 1].plot(x_axis, schemeFromm.x[0], linestyle="dashed", color="tab:red", linewidth='.8')
            axs[0, 2].plot(x_axis, schemeVanLeer.x[0], linestyle="dashed", color="tab:red", linewidth='.8')

            axs[0, 0].scatter(x_axis, schemeUpwind.x[1], marker=".", color="b")
            axs[0, 1].scatter(x_axis, schemeFromm.x[1], marker=".", color="b")
            axs[0, 2].scatter(x_axis, schemeVanLeer.x[1], marker=".", color="b",label="numerical p")
            axs[0, 0].plot(x_axis, schemeUpwind.x[1], linestyle="dashed", color="tab:blue", linewidth='.8')
            axs[0, 1].plot(x_axis, schemeFromm.x[1], linestyle="dashed", color="tab:blue", linewidth='.8')
            axs[0, 2].plot(x_axis, schemeVanLeer.x[1], linestyle="dashed", color="tab:blue", linewidth='.8')

            axs[0, 0].plot(x_axis_analytical, analytical_u, linestyle="dashed")
            axs[0, 1].plot(x_axis_analytical, analytical_u, linestyle="dashed")
            axs[0, 2].plot(x_axis_analytical, analytical_u, linestyle="dashed", label="analytical u")

            axs[0, 0].plot(x_axis_analytical, analytical_p, linestyle="dashed")
            axs[0, 1].plot(x_axis_analytical, analytical_p, linestyle="dashed")
            axs[0, 2].plot(x_axis_analytical, analytical_p, linestyle="dashed", label="analytical p")
            axs[0, 2].legend(loc="upper right", borderpad=.1)
        if i == 2:
            axs[1, 0].scatter(x_axis, schemeUpwind.x[0], marker=".", color="r")
            axs[1, 1].scatter(x_axis, schemeFromm.x[0], marker=".", color="r")
            axs[1, 2].scatter(x_axis, schemeVanLeer.x[0], marker=".", color="r", label="numerical u")
            axs[1, 0].plot(x_axis, schemeUpwind.x[0], linestyle="dashed", color="tab:red", linewidth='.8')
            axs[1, 1].plot(x_axis, schemeFromm.x[0], linestyle="dashed", color="tab:red", linewidth='.8')
            axs[1, 2].plot(x_axis, schemeVanLeer.x[0], linestyle="dashed", color="tab:red", linewidth='.8')

            axs[1, 0].scatter(x_axis, schemeUpwind.x[1], marker=".", color="b")
            axs[1, 1].scatter(x_axis, schemeFromm.x[1], marker=".", color="b")
            axs[1, 2].scatter(x_axis, schemeVanLeer.x[1], marker=".", color="b",label="numerical p")
            axs[1, 0].plot(x_axis, schemeUpwind.x[1], linestyle="dashed", color="tab:blue", linewidth='.8')
            axs[1, 1].plot(x_axis, schemeFromm.x[1], linestyle="dashed", color="tab:blue", linewidth='.8')
            axs[1, 2].plot(x_axis, schemeVanLeer.x[1], linestyle="dashed", color="tab:blue", linewidth='.8')

            axs[1, 0].plot(x_axis_analytical, analytical_u, linestyle="dashed")
            axs[1, 1].plot(x_axis_analytical, analytical_u, linestyle="dashed")
            axs[1, 2].plot(x_axis_analytical, analytical_u, linestyle="dashed")

            axs[1, 0].plot(x_axis_analytical, analytical_p, linestyle="dashed")
            axs[1, 1].plot(x_axis_analytical, analytical_p, linestyle="dashed")
            axs[1, 2].plot(x_axis_analytical, analytical_p, linestyle="dashed")
        if i == 5:
            axs[2, 0].scatter(x_axis, schemeUpwind.x[0], marker=".", color="r")
            axs[2, 1].scatter(x_axis, schemeFromm.x[0], marker=".", color="r")
            axs[2, 2].scatter(x_axis, schemeVanLeer.x[0], marker=".", color="r", label="numerical u")
            axs[2, 0].plot(x_axis, schemeUpwind.x[0], linestyle="dashed", color="tab:red", linewidth='.8')
            axs[2, 1].plot(x_axis, schemeFromm.x[0], linestyle="dashed", color="tab:red", linewidth='.8')
            axs[2, 2].plot(x_axis, schemeVanLeer.x[0], linestyle="dashed", color="tab:red", linewidth='.8')

            axs[2, 0].scatter(x_axis, schemeUpwind.x[1], marker=".", color="b")
            axs[2, 1].scatter(x_axis, schemeFromm.x[1], marker=".", color="b")
            axs[2, 2].scatter(x_axis, schemeVanLeer.x[1], marker=".", color="b",label="numerical p")
            axs[2, 0].plot(x_axis, schemeUpwind.x[1], linestyle="dashed", color="tab:blue", linewidth='.8')
            axs[2, 1].plot(x_axis, schemeFromm.x[1], linestyle="dashed", color="tab:blue", linewidth='.8')
            axs[2, 2].plot(x_axis, schemeVanLeer.x[1], linestyle="dashed", color="tab:blue", linewidth='.8')

            axs[2, 0].plot(x_axis_analytical, analytical_u, linestyle="dashed")
            axs[2, 1].plot(x_axis_analytical, analytical_u, linestyle="dashed")
            axs[2, 2].plot(x_axis_analytical, analytical_u, linestyle="dashed")

            axs[2, 0].plot(x_axis_analytical, analytical_p, linestyle="dashed")
            axs[2, 1].plot(x_axis_analytical, analytical_p, linestyle="dashed")
            axs[2, 2].plot(x_axis_analytical, analytical_p, linestyle="dashed")

        for scheme in my_twoD_schemes:
            scheme.March()

    for ax1 in axs:
        for ax in ax1:
            ax.grid()
    #         ax.set_ylim(-0.2, 1.2)

    cols = ["Upwind", "Fromm", "van Leer"]
    rows = ["N = 1", "N = 2", "N = 5"]

    plt.setp(axs[2,:], xlabel='X')
    plt.setp(axs[:,0], ylabel='q')

    pad = 5  # in points

    for ax, col in zip(axs[0], cols):
        ax.annotate(col, xy=(0.5, 1), xytext=(0, pad),
                    xycoords='axes fraction', textcoords='offset points',
                    size='large', ha='center', va='baseline')

    for ax, row in zip(axs[:, 0], rows):
        ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
                    xycoords=ax.yaxis.label, textcoords='offset points',
                    size='large', ha='right', va='center')

    fig.tight_layout()
    # tight_layout doesn't take these labels into account. We'll need
    # to make some room. These numbers are are manually tweaked.
    # You could automatically calculate them, but it's a pain.
    fig.subplots_adjust(left=0.15, top=0.95)
    plt.show()
    return 0


# ExersiseOne()
CFL = 0.4
CFL_ = "1_0"
N = 80
dx_ = 10/N
dt = CFL*dx_
dx = dt/CFL

x_axis = np.arange(dx/2, 10, dx)
x_axis_analytical = np.arange(dx/20, 10, dx/10)

print(len(x_axis))
t_axis = np.arange(0, 5, dt)
ExerciseFour1D(x_axis, x_axis_analytical, t_axis,CFL_)
