import os
import numpy as np
import gymnasium as gym

np.random.seed(1)
# Get the current working directory
current_directory = os.getcwd()

# Specify the file name
file_name = 'scripts/07_17_23:03.npy'

# Combine the current directory and file name to get the file path
file_path = os.path.join(current_directory, file_name)

# Print the file path
print(file_path)



if os.path.exists(file_path):
    print("File exists!")
else:
    print("File does not exist.")


# Load the NumPy file
data = np.load(file_path, allow_pickle=True)

# Use the data as needed


env = gym.make('highway-v0', render_mode='rgb_array')
env.configure({
    "manual_control": False,
    "observation": {
        "type": "Kinematics",
        "vehicles_count": 3,
        "features": ["car_id","presence", "x", "y", "vx", "vy"],
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

obs, info = env.reset(seed = 1) #collect a single episode (replay for later)
print("OBS: ", obs)
for d in data:
   action = d["action"]
   print("act", action)
   obs, reward, done, truncated, info = env.step(action)
   print("obs", obs)
   obs = env.render()
   
env.close()

