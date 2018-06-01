from gym.envs import register
import DeepLineWars._gym

for env in [x for x in dir(DeepLineWars._gym) if "DeepLineWars" in x]:
    clazz_env = getattr(DeepLineWars._gym, env)
    register(
        id=clazz_env.id,
        entry_point='gym_deeplinewars.envs:%s' % env
    )
