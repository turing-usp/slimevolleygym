#!/usr/bin/env python3

# Simple self-play PPO trainer

import os
import gym
import slimevolleygym
import numpy as np

from stable_baselines3.ppo import PPO
# from stable_baselines3 import logger
from stable_baselines3.common.callbacks import EvalCallback

from shutil import copyfile # keep track of generations

# Settings
SEED = 17
NUM_TIMESTEPS = int(1e9)
EVAL_FREQ = int(1e4)
EVAL_EPISODES = int(2e1)
BEST_THRESHOLD = 0.3 # must achieve a mean score above this to replace prev best self

RENDER_MODE = False # set this to false if you plan on running for full 1000 trials.

LOGDIR = "log_dir"

class SlimeVolleySelfPlayEnv(slimevolleygym.SlimeVolleyEnv):
  # wrapper over the normal single player env, but loads the best self play model
  def __init__(self):
    super(SlimeVolleySelfPlayEnv, self).__init__()
    self.policy = self
    self.best_model = None
    self.best_model_filename = None
  def predict(self, obs): # the policy
    if self.best_model is None:
      return self.action_space.sample() # return a random action
    else:
      action, _ = self.best_model.predict(obs)
      return action
  def reset(self):
    # load model if it's there
    modellist = [f for f in os.listdir(LOGDIR) if f.startswith("history")]
    modellist.sort()
    if len(modellist) > 0:
      filename = os.path.join(LOGDIR, modellist[-1]) # the latest best model
      if filename != self.best_model_filename:
        print("loading model: ", filename)
        self.best_model_filename = filename
        if self.best_model is not None:
          del self.best_model
        self.best_model = PPO.load(filename, env=self)
    return super(SlimeVolleySelfPlayEnv, self).reset()

class SelfPlayCallback(EvalCallback):
  # hacked it to only save new version of best model if beats prev self by BEST_THRESHOLD score
  # after saving model, resets the best score to be BEST_THRESHOLD
  def __init__(self, *args, **kwargs):
    super(SelfPlayCallback, self).__init__(*args, **kwargs)
    self.best_mean_reward = BEST_THRESHOLD
    self.generation = 0
  def _on_step(self) -> bool:
    result = super(SelfPlayCallback, self)._on_step()
    if result and self.best_mean_reward > BEST_THRESHOLD:
      self.generation += 1
      print("SELFPLAY: mean_reward achieved:", self.best_mean_reward)
      print("SELFPLAY: new best model, bumping up generation to", self.generation)
      source_file = os.path.join(LOGDIR, "best_model.zip")
      backup_file = os.path.join(LOGDIR, "history_"+str(self.generation).zfill(8)+".zip")
      copyfile(source_file, backup_file)
      self.best_mean_reward = BEST_THRESHOLD
    return result

def rollout(env, policy):
  """ play one agent vs the other in modified gym-style loop. """
  obs = env.reset()

  done = False
  total_reward = 0

  while not done:

    action, _states = policy.predict(obs)
    obs, reward, done, _ = env.step(action)

    total_reward += reward

    if RENDER_MODE:
      env.render()

  return total_reward

def train():
  # train selfplay agent
  # logger.configure(folder=LOGDIR)

  env = SlimeVolleySelfPlayEnv()
  env.seed(SEED)

  # take mujoco hyperparams (but doubled timesteps_per_actorbatch to cover more steps.)
  model = PPO('MlpPolicy', env, n_steps=512, batch_size=32, ent_coef=0.005, n_epochs=5,
                   learning_rate=3e-4, clip_range=0.2, gamma=0.99, gae_lambda=0.95, verbose=2)

  eval_callback = SelfPlayCallback(env,
    best_model_save_path=LOGDIR,
    log_path=LOGDIR,
    eval_freq=EVAL_FREQ,
    n_eval_episodes=EVAL_EPISODES,
    deterministic=False)

  model.learn(total_timesteps=NUM_TIMESTEPS, callback=eval_callback)

  model.save(os.path.join(LOGDIR, "final_model")) # probably never get to this point.

  env.close()

if __name__=="__main__":

  train()
