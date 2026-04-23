# True dynamics (propogates orbit)
class SpacecraftSim:
    def __init__(self, intitial_state):
        self.true_state = intitial_state

    def propogate(self, dt):
        # Use a high fidelity integrator (like Runge - kutta) to move spacecraft by dt
        return