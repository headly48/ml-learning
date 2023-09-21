# from pettingzoo.butterfly import pistonball_v4
# env = pistonball_v4.env()
# env.reset()

# for agent in env.agent_iter():
#     observation, reward, done, info = env.last()
#     action = policy(observation, agent)
#     env.step(action)


from stable_baselines3.ppo import CnnPolicy
from stable_baselines3 import PPO
from pettingzoo.butterfly import pistonball_v4
from pettingzoo.utils import random_demo

import supersuit as ss


# env = pistonball_v4.env()
# env = ss.color_reduction_v0(env, mode='B')
# env = ss.resize_v0(env, x_size=84, y_size=84)
# env = ss.frame_stack_v1(env, 3)

# # random_demo(env, render=True, episodes=1)


# model = PPO.load('policy')

# env.reset()

# for agent in env.agent_iter():
#    obs, reward, done, info = env.last()
#    act = model.predict(obs, deterministic=True)[0] if not done else None
#    env.step(act)
#    image = env.render('rgb_array')
#    print(image)




# import cv2
import numpy as np
from PIL import Image


env = pistonball_v4.env()
env = ss.color_reduction_v0(env, mode='B')
env = ss.resize_v0(env, x_size=84, y_size=84)
env = ss.frame_stack_v1(env, 3)

# random_demo(env, render=True, episodes=1)


model = PPO.load('policy-v2')

env.reset()

# videodims = (84,84)
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')    
# video = cv2.VideoWriter("test.mp4",fourcc, 60,videodims)
gif = []
for agent in env.agent_iter():
   obs, reward, done, info = env.last()
   act = model.predict(obs, deterministic=True)[0] if not done else None
   env.step(act)
   rgbImage = env.render('rgb_array')

   image = Image.fromarray(rgbImage)
   gif.append(image)
#    image.resize(videodims)

gif[0].save('temp_result_v2.gif', save_all=True,optimize=False, append_images=gif[1:], loop=0)

#    video.write(cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR))

# video.release()