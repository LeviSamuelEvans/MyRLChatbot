# src/utils.py
import os

def save_model(model, path):
    if not os.path.exists(path):
        os.makedirs(path)
    model.save(path)

def load_model(path):
    return PPO.load(path)
