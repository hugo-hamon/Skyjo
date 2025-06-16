import sys
import os

# Add src/ to the path to import the skyjo_env.py file
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.skyjo_env import SkyjoEnv
import random

env = SkyjoEnv(num_players=4)


obs = env.reset()
done = False

while not done:
    env.render()
    player_id = obs["current_player"]
    player = obs["players"][player_id]

    # Draw a card from the deck or discard pile
    random_draw = random.choice(["deck", "discard"])
    obs, _, done, info = env.step({"draw": random_draw})
    if done:
        break

    pending = obs["pending_card"]

    # Valid cards
    hidden = [
        i
        for i, v in enumerate(player["visible"])
        if not v and player["grid"][i] is not None
    ]
    visible = [
        i
        for i, v in enumerate(player["visible"])
        if v and player["grid"][i] is not None
    ]

    # Simple strategy
    # If the drawn card is weak (<= 3), replace a random visible card
    # Otherwise, flip a random hidden card
    if pending <= 3 and visible or obs["previous_action"] == "draw_discard":
        replace_index = random.choice(visible)
        obs, _, done, _ = env.step({"replace": replace_index})
    elif hidden:
        flip_index = random.choice(hidden)
        obs, _, done, _ = env.step({"flip": flip_index})
    else:
        raise ValueError("No valid cards to replace or flip")

# Final render
print("Final render\n")
env.render()
