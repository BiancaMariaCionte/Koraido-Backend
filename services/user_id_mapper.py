import json
import os
import pandas as pd

USER_MAP_PATH = 'ml-latest-small/user_id_map.json'

def load_user_map():
    if os.path.exists(USER_MAP_PATH):
        with open(USER_MAP_PATH, 'r') as f:
            return json.load(f)
    return {}

def save_user_map(user_map):
    with open(USER_MAP_PATH, 'w') as f:
        json.dump(user_map, f)


def get_numeric_user_id(firebase_uid):
    user_map = load_user_map()

    # Return if already mapped
    if firebase_uid in user_map:
        return user_map[firebase_uid]

    # Otherwise, generate new numeric ID based on the existing user map
    if user_map:
        last_user_id = max(user_map.values())
        new_user_id = int(last_user_id) + 1
    else:
        new_user_id = 1200

    # Save the mapping
    user_map[firebase_uid] = new_user_id
    save_user_map(user_map)

    return new_user_id
