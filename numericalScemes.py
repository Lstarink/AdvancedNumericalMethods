import numpy as np
import matplotlib.pyplot as plt

class NumericalScheme:
    def __init__(self, state0, spatial_axis, time_axis, u, scheme="Upwind"):
        self.state0 = state0
        self.state = state0
        self.spatial_axis = spatial_axis
        self.time_axis = time_axis
        self.u = u
        self.mesh = np.meshgrid(spatial_axis, time_axis)
        self.scheme = scheme

    def TickUpwind(self):
        state_plus = np.zeros(self.state.shape)
        for n, state_n in enumerate(self.state):
            region_of_interest = self.Wrapp(n)
            # print(region_of_interest)
            # print(self.state)
            [dt, dx] = self.StepSize(n)
            state_plus[n] = self.Upwind(region_of_interest, dt, dx)
        self.state = state_plus

    def TickFromm(self):
        state_plus = np.zeros(self.state.shape)
        for n, state_n in enumerate(self.state):
            region_of_interest = self.Wrapp(n, scope=2)
            # print(region_of_interest)
            # print(self.state)
            [dt, dx] = self.StepSize(n)
            state_plus[n] = self.Fromm(region_of_interest, dt, dx)
        self.state = state_plus

    def Upwind(self, region_of_interest, dt, dx):
        if self.u < 0:
            region_of_interest = np.flip(region_of_interest)
        q_plus = region_of_interest[1] - abs(self.u)*dt*(region_of_interest[1]-region_of_interest[0])/dx
        return q_plus

    def Fromm(self, region_of_interest, dt, dx):
        if self.u < 0:
            region_of_interest = np.flip(region_of_interest)

        Qi_min2 = region_of_interest[0]
        Qi_min1 = region_of_interest[1]
        Qi = region_of_interest[2]
        Qi_plus1 = region_of_interest[3]
        q_plus = Qi - (0.25*abs(self.u)*dt/dx)*(Qi_plus1 + 3*Qi - 5*Qi_min1 + Qi_min2) + \
                 (0.25*(abs(self.u)*dt/dx)**2)*(Qi_plus1 - Qi - Qi_min1 + Qi_min2)

        return q_plus

    def TickVanLeer(self):
        state_plus = np.zeros(self.state.shape)
        for n, state_n in enumerate(self.state):
            region_of_interest = self.Wrapp(n, scope=2)
            # print(region_of_interest)
            # print(self.state)
            [dt, dx] = self.StepSize(n)
            state_plus[n] = self.VanLeer(region_of_interest, dt, dx)
        self.state = state_plus
        return 0

    def VanLeer(self, region_of_interest, dt, dx):
        nu = self.u*dt/dx
        if self.u <0:
            region_of_interest = np.flip(region_of_interest)

        Qi_min2 = region_of_interest[0]
        Qi_min1 = region_of_interest[1]
        Qi = region_of_interest[2]
        Qi_plus1 = region_of_interest[3]
        Qi_plus2 = region_of_interest[4]

        theta_i_min_half = (Qi-Qi_min1)/(Qi_min1-Qi_min2)
        theta_i_plus_half = (Qi-Qi_plus1)/(Qi_plus1-Qi_plus2)

        phi_i_min_half = (theta_i_min_half + abs(theta_i_min_half))/(1 + abs(theta_i_min_half))
        phi_i_plus_half = (theta_i_plus_half + abs(theta_i_plus_half))/(1 + abs(theta_i_plus_half))

        Q_plus = Qi - nu*(Qi - Qi_min1) - 0.5*nu*(1-nu)*(phi_i_plus_half*(Qi_plus1-Qi) - phi_i_min_half*(Qi-Qi_min1))
        return Q_plus

    def Wrapp(self, n, scope=1):
        state = self.state
        #For periodic bounds
        if n >= len(state)-scope:
            state = np.roll(state, len(state)-scope-n-1)
            region_of_interest = state[-(scope*2+1):]
        elif n < scope:
            state = np.roll(state, scope-n)
            region_of_interest = state[0:scope*2+1]
        else:
            region_of_interest = state[n-scope:n+scope+1]
        return region_of_interest

    def StepSize(self, n):
        if n != 0:
            dx = self.spatial_axis[n] - self.spatial_axis[n-1]
        else:
            dx = self.spatial_axis[n+1] - self.spatial_axis[n] # Not sure
        dt = self.time_axis[1]-self.time_axis[0]
        return [dt, dx]

    def PlotX(self):
        plt.figure()
        plt.step(self.spatial_axis, self.state0)
        plt.step(self.spatial_axis, self.state)
        plt.show()
