from gym import wrappers

def run_agent(agent, env):
    outdir = 'random-agent-results'
    env = wrappers.Monitor(env, directory=outdir, force=True)
    env.seed(0)
    episode_count = 10
    reward = 0
    done = False
    for _ in range(episode_count):
        ob = env.reset()
        while True:
            action = agent.act(ob, reward, done)
            ob, reward, done, _ = env.step(action)
            if done:
                break
    env.close()
