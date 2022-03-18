import gym
import pickle
import torch
# import pickle5
import slimevolleygym
from stable_baselines3 import PPO

if __name__=="__main__":
    env = gym.make("SlimeVolley-v0")

    with open('model_dict.pickle', 'rb') as handle:
        model_dict = pickle.load(handle)

    model_path = 'log_dir/nelson_new_model'

    model = PPO('MlpPolicy', env)
    model.set_parameters(model_dict, device=torch.device('cpu'))

    print("Saving", model_path)
    model.save(model_path)