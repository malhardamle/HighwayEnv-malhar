import pygame
import time, sys
sys.path.append("/Users/malhardamle/Desktop/HighwayEnv-malhar/highway_env/envs/common")

from highway_env.envs.common import test2


print(test2.var)

# def main():
#     pygame.init()

#     key_mapping = {
#         pygame.K_UP: "up",
#         pygame.K_DOWN: "down",
#         pygame.K_LEFT: "left",
#         pygame.K_RIGHT: "right"
#     }

#     arrow_key_list = []

#     start_time = time.time()

#     while time.time() - start_time < 10:
#         for event in pygame.event.get():
#             if event.type == pygame.KEYDOWN and event.key in key_mapping:
#                 key_name = key_mapping[event.key]
#                 arrow_key_list.append(key_name)

#     pygame.quit()

#     print("Captured Arrow Keys:")
#     for i, key_name in enumerate(arrow_key_list, start=1):
#         print(f"{i}. {key_name}")

# if __name__ == "__main__":
#     main()
