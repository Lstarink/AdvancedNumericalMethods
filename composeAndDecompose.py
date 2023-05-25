import numpy as np
import numericalScemes

class ComposeDecompose:
    def __init__(self, A, x0, x_grid, time_grid):
        assert(len(A) == len(x0))
        self.spatial_axis = x_grid
        self.time_axis = time_grid
        self.x = x0
        self.A = A
        self.dimension = len(A)
        self.R, self.R_inv, self.Phi, self.speeds = ComposeDecompose.Diagonalize(self)
        self.w = ComposeDecompose.Decompose(self)
        self.schemes = ComposeDecompose.SetupSchemes(self)
        print("schemes: ", self.schemes)


    def Diagonalize(self):
        eigenvalues, eigenvectors= np.linalg.eig(self.A)
        is_real = np.isreal(eigenvalues)
        print(is_real)
        for real_eigenvalue in is_real:
            assert(real_eigenvalue)

        print(eigenvalues)
        print(eigenvectors)
        R = eigenvectors
        R_inv = np.linalg.inv(R)
        Phi = np.diag(eigenvalues)
        print("A: ", self.A)
        print("Reconstructed A:", R@Phi@R_inv)
        return R, R_inv, Phi, eigenvalues

    def Decompose(self):
        w = np.zeros(self.x.shape)

        for n, x_n in enumerate(self.x.T):
            w[:, n] = self.R@x_n
        return w

    def Compose(self):
        x = np.zeros(self.x.shape)

        for n, w_n in enumerate(self.w.T):
            x[:, n] = self.R_inv @ w_n

        return x

    def SetupSchemes(self):
        schemes = []

        for n in range(self.dimension):
            scheme = numericalScemes.NumericalScheme(self.w[n], self.spatial_axis, self.time_axis, self.speeds[n])
            schemes.append(scheme)

        return schemes

    def March(self):
        for n, scheme in enumerate(self.schemes):
            scheme.TickFromm()
            # scheme.PlotX()
            self.w[n] = scheme.state
        self.x = self.Compose()

if __name__ == "__main__":
    cd = ComposeDecompose(np.array([[0, 2],[2, 0]]), np.zeros([2,8]), np.linspace(0, 10, 100), np.linspace(0, 1, 10))