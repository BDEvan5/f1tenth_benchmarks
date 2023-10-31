
def run_simulation_loop_laps(env, planner, n_laps):
    for lap in range(n_laps):
        observation, done, init_pose = env.reset()
        while not done:
            action = planner.plan(observation)
            observation, done = env.step(action)
    env.save_data_frame()


def run_training_loop_steps(env, planner, steps):
    observation, done, init_pose = env.reset()
    
    for i in range(steps):
        action = planner.plan(observation)
        observation, done = env.step(action)
        if done:
            planner.done_callback(observation)
            observation, done, init_pose = env.reset()