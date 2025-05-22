# import os
# import pandas as pd
# from services.firebase_config import db
# from services.user_id_mapper import get_numeric_user_id

# def generate_user_info_xlsx(firebase_uid):
#     user_id = get_numeric_user_id(firebase_uid)
#     print(f'Generated numeric user ID: {user_id} for Firebase UID: {firebase_uid}')

#     # Fetch user data from Firestore
#     user_ref = db.collection('users').document(firebase_uid)
#     user_doc = user_ref.get()

#     if not user_doc.exists:
#         print(f'User {firebase_uid} not found in Firebase')
#         return

#     user_data = user_doc.to_dict()

#     # Exclude gender and country from the user data
#     user_data_filtered = {key: value for key, value in user_data.items() if key not in ['gender', 'country']}
    
#     # Ensure we are still getting the required fields (e.g., interests) from the filtered data
#     interests = '|'.join(user_data_filtered.get('interests', []))
#     dummy_age = 30  # Replace with actual logic if needed

#     file_path = 'ml-latest-small/user_info.xlsx'

#     # Ensure the directory exists
#     directory = os.path.dirname(file_path)
#     if not os.path.exists(directory):
#         os.makedirs(directory)

#     try:
#         # Read the existing Excel file
#         df_existing = pd.read_excel(file_path)
#     except FileNotFoundError:
#         # If file doesn't exist, create an empty DataFrame with the correct columns
#         df_existing = pd.DataFrame(columns=['userId', 'age', 'interests'])

#     # Check if the user already exists by user_id
#     if user_id in df_existing['userId'].values:
#         #print("Existing user IDs in user_info.xlsx:", df_existing['userId'].tolist())
#         print(f'user_info.xlsx: user {firebase_uid} already exists')

#         # Update the interests for the existing user
#         df_existing.loc[df_existing['userId'] == user_id, 'interests'] = interests
#     else:
#         # Create new data for the user if they don't exist
#         df_new = pd.DataFrame([{
#             'userId': user_id,
#             'age': dummy_age,
#             'interests': interests
#         }])

#         # Concatenate the new user data with the existing data
#         df_existing = pd.concat([df_existing, df_new], ignore_index=True)

#     # Debugging: Print the result DataFrame
#     print(f"Data to be saved to Excel: {df_existing}")

#     # Save the updated data back to the Excel file
#     with pd.ExcelWriter(file_path, engine='openpyxl', mode='w') as writer:
#         df_existing.to_excel(writer, index=False)

#     print(f'🥽 user_info.xlsx: updated user {firebase_uid} -> {user_id}')

from services.firebase_config import db
from services.user_id_mapper import get_numeric_user_id

def generate_user_info_xlsx(firebase_uid):
    user_id = get_numeric_user_id(firebase_uid)
    print(f'Generated numeric user ID: {user_id} for Firebase UID: {firebase_uid}')

    # Fetch user data from Firestore
    user_ref = db.collection('users').document(firebase_uid)
    user_doc = user_ref.get()

    if not user_doc.exists:
        print(f'User {firebase_uid} not found in Firebase')
        return

    user_data = user_doc.to_dict()
    interests = '|'.join(user_data.get('interests', []))
    dummy_age = 30  # Replace with actual logic if needed

    user_info = {
        'userId': user_id,
        'age': dummy_age,
        'interests': interests
    }

    user_info_ref = db.collection('user_info').document(str(user_id))
    user_info_ref.set(user_info)

    print(f'✅ Firestore user_info updated for Firebase UID {firebase_uid} (user ID {user_id})')

