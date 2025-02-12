import metaworld
import numpy as np

mt1 = metaworld.MT1('drawer-close-v2')
env = mt1.train_classes['drawer-close-v2']()
task = mt1.train_tasks[0]
env.set_task(task)

obs = env.reset()
env.render_mode = "human"  # Set render mode explicitly

for _ in range(1000):
    action = np.random.uniform(-1, 1, env.action_space.shape)
    result = env.step(action)  # Get all returned values
    obs = result[0]  # Unpack only observation
    env.render()