# Handles SPICE, planet positions, time
class Universe:
    def get_planet_position(self, planet_name, time_et): # Returns 3x1 vector of planet position
        return