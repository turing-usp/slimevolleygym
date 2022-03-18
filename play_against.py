"""
Simple evaluation example.

run: python eval_ppo.py --render

Evaluate PPO1 policy (MLP input_dim x 64 x 64 x output_dim policy) against built-in AI

"""

import warnings
# numpy warnings because of tensorflow
warnings.filterwarnings("ignore", category=FutureWarning, module='tensorflow')
warnings.filterwarnings("ignore", category=UserWarning, module='gym')

import gym
import numpy as np
import argparse

import slimevolleygym
from stable_baselines3 import PPO

RENDER_MODE = True

class SlimeVolleyVersusEnv(slimevolleygym.SlimeVolleyEnv):
  # wrapper over the normal single player env, but loads the best self play model
  def __init__(self, model_left, model_right):
    super(SlimeVolleyVersusEnv, self).__init__()
    self.policy = self
    self.model_left = model_left
    self.model_right = model_right
  def predict(self, obs): # the policy
    if self.model_right is None:
      return self.action_space.sample() # return a random action
    else:
      action, _ = self.model_left.predict(obs, deterministic=True)
      return action
  def step(self, action):
      return super(SlimeVolleyVersusEnv, self).step(action)

def rollout(env, render_mode=False):
  """ play one agent vs the other in modified gym-style loop. """
  obs = env.reset()

  done = False
  total_reward = 0

  while not done:

    # action, _states = policy.predict(obs, deterministic=True)
    obs, reward, done, _ = env.step(0)

    total_reward += reward

    if render_mode:
      env.render()

  return total_reward

if __name__=="__main__":
    if RENDER_MODE:
        from pyglet.window import key
        from time import sleep

    manualAction = [0, 0, 0] # forward, backward, jump
    otherManualAction = [0, 0, 0]
    manualMode = False
    otherManualMode = False

    # taken from https://github.com/openai/gym/blob/master/gym/envs/box2d/car_racing.py
    def key_press(k, mod):
        global manualMode, manualAction, otherManualMode, otherManualAction
        if k == key.LEFT:  manualAction[0] = 1
        if k == key.RIGHT: manualAction[1] = 1
        if k == key.UP:    manualAction[2] = 1
        if (k == key.LEFT or k == key.RIGHT or k == key.UP): manualMode = True

        if k == key.D:     otherManualAction[0] = 1
        if k == key.A:     otherManualAction[1] = 1
        if k == key.W:     otherManualAction[2] = 1
        if (k == key.D or k == key.A or k == key.W): otherManualMode = True

    def key_release(k, mod):
        global manualMode, manualAction, otherManualMode, otherManualAction
        if k == key.LEFT:  manualAction[0] = 0
        if k == key.RIGHT: manualAction[1] = 0
        if k == key.UP:    manualAction[2] = 0
        if k == key.D:     otherManualAction[0] = 0
        if k == key.A:     otherManualAction[1] = 0
        if k == key.W:     otherManualAction[2] = 0


    parser = argparse.ArgumentParser(description='Evaluate pre-trained PPO agent.')
    parser.add_argument('--left-model-path', help='path to stable-baselines model.',
                            type=str, default="log_dir/new_luis_model")
    parser.add_argument('--right-model-path', help='path to stable-baselines model.',
                            type=str, default="log_dir/new_luis_model")
    parser.add_argument('--render', action='store_true', help='render to screen?', default=True)

    args = parser.parse_args()
    render_mode = args.render

    env = gym.make("SlimeVolley-v0")

    # the yellow agent:
    print("Loading", args.left_model_path)
    left_model = PPO.load(args.left_model_path, env=env)
    # the blue agent:
    print("Loading", args.right_model_path)
    right_model = PPO.load(args.right_model_path, env=env)

    null_model = PPO('MlpPolicy', env)

    env = SlimeVolleyVersusEnv(model_left=left_model, model_right=right_model)

    if RENDER_MODE:
        env.render()
        env.viewer.window.on_key_press = key_press
        env.viewer.window.on_key_release = key_release
    
    while True:
        obs = env.reset()
        done = False
        manualMode = False
        while not done:
            if manualMode: # override with keyboard
                action = manualAction
            else:
                action, _ = right_model.predict(obs)

            obs, reward, done, _ = env.step(action)                

            if RENDER_MODE:
                env.render()
                sleep(0.005) # 0.01

