import gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from enviroment import ChatbotEnv

def train_model():
    """Train a PPO model on the ChatbotEnv environment."""
    env = make_vec_env(lambda: ChatbotEnv(), n_envs=1)
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=10000)
    model.save("models/saved_model/ppo_chatbot")

if __name__ == "__main__":
    train_model()
