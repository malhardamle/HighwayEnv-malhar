import gymnasium as gym
import numpy as np 
from datetime import datetime

output = datetime.now().strftime("%Y_%m_%d%H:%M:%S")
output = output + ".npy"
env = gym.make('highway-v0', render_mode='human')
env.configure({
    "manual_control": True,
    "observation": {
        "type": "Kinematics",
        "vehicles_count": 3,
        "features": ["presence", "x", "y", "vx", "vy"],
        "features_range": {
            "x": [-100, 100],
            "y": [-100, 100],
            "vx": [-20, 50],
            "vy": [-20, 50]
        },
        "absolute": False,
        "order": "sorted"
    }
})
print(env.config)
obs, info = env.reset()