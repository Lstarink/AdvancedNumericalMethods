import numpy as np

class NumericalSimulation:
    def __init__(self, state0, spatial_axis, time_axis, u):
        self.state = state0
        self.spatial_axis = spatial_axis
        self.time_axis = time_axis
        self.u = u
        self.mesh = np.meshgrid(spatial_axis, time_axis)
        self.solution = np.zeros([len(spatial_axis), len(time_axis)])

    def Tick(self):
        state_plus = np.zeros(self.state.shape)
        for n, state_n in enumerate(self.state):
            region_of_interest = self.Wrapp(n)
            print(region_of_interest)
            print(self.state)
            [dt, dx] = self.StepSize(n)
            state_plus[n] = self.TickUpwind(region_of_interest, dt, dx)
        self.state = state_plus

    def TickUpwind(self, region_of_interest, dt, dx):
        if self.u < 0:
            region_of_interest = np.flip(region_of_interest)
        q_plus = region_of_interest[1] - self.u*dt*(region_of_interest[1]-region_of_interest[0])/dx
        return q_plus

    def TickFromm(self):
        return 0

    def TickVanLeer(self):
        return 0

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
