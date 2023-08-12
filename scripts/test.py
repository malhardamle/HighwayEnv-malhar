from pynput import keyboard

# Define the actions dictionary
ACTIONS_ALL = {
    0: 'LANE_LEFT',
    1: 'IDLE',
    2: 'LANE_RIGHT',
    3: 'FASTER',
    4: 'SLOWER'
}

recorded_actions = []

def on_press(key):
    try:
        if key.char == 'q':
            return False  # Stop the listener when 'q' is pressed

    except AttributeError:
        if key == keyboard.Key.up:
            recorded_actions.append(ACTIONS_ALL[3])
        elif key == keyboard.Key.down:
            recorded_actions.append(ACTIONS_ALL[4])
        elif key == keyboard.Key.left:
            recorded_actions.append(ACTIONS_ALL[0])
        elif key == keyboard.Key.right:
            recorded_actions.append(ACTIONS_ALL[2])

def record_arrow_keystrokes():
    global recorded_actions

    print("Press arrow keys to record actions. Press 'q' to stop recording.")
    with keyboard.Listener(on_press=on_press) as listener:
        listener.join()

if __name__ == "__main__":
    record_arrow_keystrokes()
    print("Recorded Actions:", recorded_actions)
