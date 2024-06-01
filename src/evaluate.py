# src/evaluate.py
from stable_baselines3 import PPO
from transformers import GPT2Tokenizer
from enviroment import ChatbotEnv

def evaluate_chatbot(model_path, num_episodes=10):
    """Evaluate a chatbot model using the PPO algorithm."""
    model = PPO.load(model_path)
    env = ChatbotEnv()
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    for episode in range(num_episodes):
        obs = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs)
            obs, reward, done, _ = env.step(action)
            print(f"Response: {tokenizer.decode(action, skip_special_tokens=True)}")

if __name__ == "__main__":
    evaluate_chatbot("models/saved_model/ppo_chatbot")
