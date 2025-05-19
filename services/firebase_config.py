import firebase_admin
from firebase_admin import credentials, firestore

# Initialize Firebase Admin SDK
cred = credentials.Certificate("../services/koraido-firebase-adminsdk1.json")

firebase_admin.initialize_app(cred)

# Initialize Firestore
db = firestore.client()
