
def run_simulation_loop_laps(env, planner, n_laps):
    for lap in range(n_laps):
        observation, done, init_pose = env.reset()
        while not done:
            action = planner.plan(observation)
            observation, done = env.step(action)
    env.save_data_frame()