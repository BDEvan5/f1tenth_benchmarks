import cProfile, io, pstats

def run_simulation_loop_laps(sim, planner, n_laps):
    pr = cProfile.Profile()
    pr.enable()
    for lap in range(n_laps):
        observation, done, init_pose = sim.reset()
        while not done:
            action = planner.plan(observation)
            observation, done = sim.step(action)
    pr.disable()
    print_stats(pr, planner.name, sim.map_name)

def print_stats(pr, name, map_name):
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    planning_time = ps.get_stats_profile().func_profiles['plan']
    print(f"Planning time: {planning_time.cumtime} seconds = {planning_time.cumtime / int(planning_time.ncalls)} seconds per call")
    with open(f"Logs/{name}/Profile_{map_name}.txt", "w") as f:
        ps.print_stats(0.2)
        f.write(s.getvalue())


def run_training_loop_steps(sim, planner, steps):
    observation, done, init_pose = sim.reset()
    
    for i in range(steps):
        action = planner.plan(observation)
        observation, done = sim.step(action)
        if done:
            planner.done_callback(observation)
            observation, done, init_pose = sim.reset()