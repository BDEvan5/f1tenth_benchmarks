import cProfile, io, pstats

def run_simulation_loop_laps(sim, planner, n_laps):

    for lap in range(n_laps):
        observation, done, init_pose = sim.reset()
        while not done:
            action = planner.plan(observation)
            observation, done = sim.step(action)



def run_training_loop_steps(sim, planner, steps):
    observation, done, init_pose = sim.reset()
    
    for i in range(steps):
        action = planner.plan(observation)
        observation, done = sim.step(action)
        if done:
            planner.done_callback(observation)
            observation, done, init_pose = sim.reset()