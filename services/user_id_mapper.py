# import json
# import os
# import pandas as pd

# USER_MAP_PATH = 'ml-latest-small/user_id_map.json'

# def load_user_map():
#     if os.path.exists(USER_MAP_PATH):
#         with open(USER_MAP_PATH, 'r') as f:
#             return json.load(f)
#     return {}

# # def save_user_map(user_map):
# #     with open(USER_MAP_PATH, 'w') as f:
# #         json.dump(user_map, f)


# def save_user_map(user_map):
#     os.makedirs(os.path.dirname(USER_MAP_PATH), exist_ok=True)  # Ensure directory exists
#     with open(USER_MAP_PATH, 'w') as f:
#         json.dump(user_map, f, indent=2)


# def get_numeric_user_id(firebase_uid):
#     user_map = load_user_map()

#     # Return if already mapped
#     if firebase_uid in user_map:
#         return user_map[firebase_uid]

#     # Otherwise, generate new numeric ID based on the existing user map
#     if user_map:
#         last_user_id = max(user_map.values())
#         new_user_id = int(last_user_id) + 1
#     else:
#         new_user_id = 1200

#     # Save the mapping
#     user_map[firebase_uid] = new_user_id
#     save_user_map(user_map)

#     return new_user_id


import json
import os
import pandas as pd
from services.firebase_config import db  # Adjust based on your setup

USER_MAP_PATH = 'ml-latest-small/user_id_map.json'

def load_user_map():
    if os.path.exists(USER_MAP_PATH):
        with open(USER_MAP_PATH, 'r') as f:
            return json.load(f)
    return {}

def save_user_map(user_map):
    os.makedirs(os.path.dirname(USER_MAP_PATH), exist_ok=True)
    with open(USER_MAP_PATH, 'w') as f:
        json.dump(user_map, f, indent=2)

def get_numeric_user_id(firebase_uid):
    user_map = load_user_map()

    # Already mapped
    if firebase_uid in user_map:
        return user_map[firebase_uid]

    # Generate new numeric ID
    if user_map:
        last_user_id = max(user_map.values())
        new_user_id = int(last_user_id) + 1
    else:
        new_user_id = 1200

    # Update local JSON
    user_map[firebase_uid] = new_user_id
    save_user_map(user_map)

    # Write to Firestore
    db.collection("userId_map").document(str(new_user_id)).set({
        "numericId": new_user_id,
        "firebaseId": firebase_uid
    })

    return new_user_id

