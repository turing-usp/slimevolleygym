import gym
import pickle
import slimevolleygym
from stable_baselines3 import PPO

if __name__=="__main__":
    env = gym.make("SlimeVolley-v0")

    model_path = 'log_dir/bernardo_new_model'

    print("Loading", model_path)
    model = PPO.load(model_path, env=env)
    model_dict = model.get_parameters()

    with open('model_dict.pickle', 'wb') as handle:
        pickle.dump(model_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)