# import firebase_admin
# from firebase_admin import credentials, firestore

# # Initialize Firebase Admin SDK
# cred = credentials.Certificate("../services/koraido-firebase-adminsdk-lq50y-ad14b6d0b5.json")

# firebase_admin.initialize_app(cred)

# # Initialize Firestore
# db = firestore.client()

import os
import json
import firebase_admin
from firebase_admin import credentials, firestore

# Load credentials from environment variable
firebase_credentials = os.environ.get("FIREBASE_CREDENTIALS_JSON")

if not firebase_credentials:
    raise RuntimeError("Missing FIREBASE_CREDENTIALS_JSON environment variable.")

# Parse JSON string into Python dict
cred_dict = json.loads(firebase_credentials)

# Create a credentials object from dict
cred = credentials.Certificate(cred_dict)

# Initialize Firebase app if not already initialized
if not firebase_admin._apps:
    firebase_admin.initialize_app(cred)

# Initialize Firestore client
db = firestore.client()
