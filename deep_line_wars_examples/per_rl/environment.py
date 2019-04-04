class Environment:

    def __init__(self, spec):
        env_spec = spec["environment"]
        self.env = env_spec["model"]
        self.env_episodes = env_spec["episodes"]

    def run(self):
        for i in range(self.env_episodes):
            self.on_episode_start()  # Call start of episode callback

            t = False
            s = self.env.reset()
            self.observe(s, 0, t)  # Initial State observation
            steps = 0
            while not t:

                a = self.act()
                s1, r, t, _ = self.env.step(a.item())
                steps += 1

            self.train()
            self.log_scalar("accumulative_reward", steps, i)  # Log cummulative reward
            self.on_episode_end()  # Call end of episode callback
