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
      obs = self.getObs()
      action, _ = self.model_right.predict(obs, deterministic=True)
    #   print(action)
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

  parser = argparse.ArgumentParser(description='Evaluate pre-trained PPO agent.')
  parser.add_argument('--left-model-path', help='path to stable-baselines model.',
                        type=str, default="log_dir/nelson_new_model")
  parser.add_argument('--right-model-path', help='path to stable-baselines model.',
                        type=str, default="log_dir/bernardo_new_model")
  parser.add_argument('--render', action='store_true', help='render to screen?', default=False)

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

  left_win = 0
  right_win = 0
  history = []
  for i in range(1, 51):
    env.seed(seed=i)
    cumulative_score = rollout(env, render_mode)
    left_win += 1 if cumulative_score < 0 else 0 
    right_win += 1 if cumulative_score > 0 else 0 
    print("cumulative score #", i, ":", cumulative_score, f" Bolinha Peluda wins: {left_win} Alan wins: {right_win}")
    history.append(cumulative_score)

  # print("history dump:", history)
  # this is what I got: [1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 2, 1, 1, 1, 1, 4, 0, 0, 0, 2, 2, 0, 1, 2, 2, 2, 1, 2, 1, 0, 2, 0, 1, 1, 1, 0, 0, 1, 0, 1, 2, 0, 0, 1, 1, 4, 1, 0, 2, 2, 3, 2, 4, 4, 1, 1, 2, 0, 0, 0, 4, 1, 1, 2, 0, 1, 1, 1, 2, 1, 1, 3, 2, 0, 1, 1, 1, 2, 2, 1, 1, 0, 0, 0, 1, 1, 1, 2, 5, 3, 3, 0, 0, 1, 0, 0, 2, 2, 1, 2, 1, 1, 0, 1, 0, 1, 1, 2, 2, 1, 3, 4, 0, 0, 0, 3, 0, 1, 5, 2, 4, 0, 1, 1, 1, 3, 0, 1, 2, 1, 1, 2, 1, 1, 2, 0, 1, 1, 0, 1, 0, 1, 2, 0, 2, 0, 2, 1, 1, 1, 0, 0, 0, 2, 2, 1, 0, 0, 0, 3, 0, 1, 3, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 2, 0, 1, 2, 1, 0, 0, 1, 2, 0, 2, 1, 0, 1, 2, 2, 0, 2, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 2, 0, 1, 1, 0, 2, 1, 0, 1, 0, 1, 0, 1, 3, 2, 2, 1, 2, 0, 2, 2, 0, 1, 0, 1, 0, 0, 2, 1, 2, 1, 0, 2, 1, 0, 1, 0, 2, 1, 1, 1, 2, 2, 2, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 2, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 2, 0, 2, 0, 0, 1, 1, 0, 1, 2, 1, 0, 2, 3, 3, 4, 0, 0, 1, 0, 1, 1, 2, 0, 1, 0, 1, 0, 2, 1, 0, 3, 0, 0, 1, 1, 1, 2, 2, 0, 0, 2, 0, 0, 1, 2, 4, 0, 2, 0, 1, 1, 1, 0, 1, 2, 1, 0, 0, 4, 1, 0, 0, 0, 0, 2, 1, 1, 1, 3, 1, 1, 1, 2, 1, 1, 1, 2, 1, 0, 1, 1, 2, 0, 0, 0, 1, 4, 2, 3, 0, 3, 1, 0, 0, 1, 2, 2, 1, 0, 0, 1, 2, 0, 2, 1, 0, 1, 0, 0, 0, 1, 0, 2, 1, 2, 0, 1, 1, 2, 1, 0, 1, 0, 1, 1, 2, 0, 2, 0, 0, 1, 1, 0, 0, 2, 0, 2, 0, 1, 2, 2, 3, 1, 1, 0, 0, 1, 1, 4, 2, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 2, 3, 0, 0, 2, 2, 0, 3, 1, 0, 2, 0, 1, 0, 0, 2, 1, 2, 3, 1, 0, 1, 0, 1, 2, 1, 0, 2, 0, 0, 1, 0, 0, 1, 1, 1, 0, 2, 1, 0, 2, 2, 0, 1, 0, 1, 0, 5, 2, 2, 0, 1, 2, 0, 2, 0, 0, 0, 1, 0, 0, 1, 0, 2, 2, 1, 0, 1, 1, 2, 0, 0, 2, 0, 0, 3, 2, 2, -1, 3, 1, 1, 2, 0, 0, 2, 1, 1, 0, 1, 1, 3, 0, 2, 1, 1, 0, 3, 2, 1, 0, 2, 1, 2, 0, 1, 0, 2, 0, 2, 0, 3, 0, 0, 1, 0, 0, 1, 0, 0, 0, 2, 1, 2, 0, 3, 0, 2, 0, 1, 2, 1, 0, 0, 1, 2, 1, 0, 0, 4, 3, 0, 2, 1, 0, 0, 0, 2, 2, 1, 1, 0, 0, 2, 1, 0, 2, 2, 1, 0, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 2, 2, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 2, 0, 0, 0, 2, 2, 2, 0, 0, 4, 3, 0, 0, 1, 0, 1, 1, 3, 3, 1, 0, 1, 1, 0, 0, 3, 3, 0, 2, 3, 1, 2, 1, 3, 2, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 2, 0, 1, 1, 2, 1, 3, 1, 2, 0, -1, 0, 1, 0, 1, 4, 4, 0, 0, 0, 1, 0, 1, 0, 1, 3, 1, 0, 1, 1, 1, 0, 1, 1, 0, 2, 0, 2, 0, 0, 2, 1, 1, 1, 0, 1, 3, 1, 0, 0, 0, 1, 1, 0, 1, 2, 0, 2, 2, 0, 1, 0, 2, 3, 1, 1, 1, 1, 0, 2, 2, 1, 2, 0, 0, 2, 0, 1, 3, 0, 1, 0, 1, 0, 1, 0, 0, 2, 1, 2, 0, 2, 1, 1, 3, 1, 2, 2, 0, 1, 0, 2, 0, 1, 2, 0, 1, 2, 1, 0, 0, 0, 2, 0, 0, 2, 0, 0, 0, 0, 0, 1, 0, 0, 1, 2, 1, 2, 0, 0, 0, 2, 1, 1, 3, 1, 2, 2, 2, 2, 0, 1, 1, 1, 2, 0, 1, 4, 0, 0, 0, 1, 4, 0, 1, 4, 1, 2, 1, 1, 3, 3, 3, 4, 1, 0, 1, 0, 0, 3, 1, 4, 1, 3, 1, 1, 1, 0, 2, 4, 1, 0, 3, 2, 1, 0, 0, 3, 1, 2, 0, 0, 0, 4, 0, 1, 0, 1, 1, 0, 0, 0, 0, 2, 1, 1, 0, 2, 3, 0, 1, 0, 1, 1, 2, 0, 1, 1, 1, 2, 1, 2, 1, 1, 1, 1, 2, 0, 0, 1, 0, 0, 1, 2, 1, 3, 2, 0, 2, 0, 0, 0, 0, 1, 0, 2, 0, 1, 0, 1, 0, 1, 0, 0, 2, 1, 1, 2, 0, 1, 1, 1, 1, 2, 1, 0, 1, 0, 2, 3, 3, 0, -1, 2, 0, 1, 1, 3, 0, 1, 0, 0, 3, 0, 2, 0, 0, 1, 0, 2, 2, -1, 1, 0, 0, 1, 0, 1, 1, 0, 2, 1, 3, 1, 0, 2, 2, 1, 1, 1, 1, 1, 3, 1, 1, 2, 0, 2, 2, 1, 0, 0, 2, 0, 1, 2, 3, 2, 3, 0, 3, 2, 3, 2]
  print("average score", np.mean(history), "standard_deviation", np.std(history))
