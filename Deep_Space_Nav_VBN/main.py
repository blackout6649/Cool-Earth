# # Pseudo-code for main.py
#
# # 1. Setup
# sim = SpacecraftSim(true_initial_state)
# fsw = NavigationFilter(noisy_initial_guess)
# camera = CameraSim()
#
# # 2. Loop over trajectory
# for t in time_steps:
#     # --- PHYSICAL WORLD STEPS ---
#     sim.propagate(dt)
#
#     # Generate a measurement (The "Truth" creates data)
#     true_relative_pos = universe.get_mars() - sim.true_state
#     measurement = camera.take_picture(true_relative_pos)  # Returns (u, v) pixels
#
#     # --- FLIGHT SOFTWARE STEPS ---
#     # The FSW predicts where it thinks it is
#     fsw.predict(dt)
#
#     # The FSW corrects its estimate using the camera data
#     fsw.update(measurement)
#
#     # --- LOGGING ---
#     save_error(sim.true_state - fsw.estimated_state)