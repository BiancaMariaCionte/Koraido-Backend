import requests

def trigger_generate_user(firebase_uid):
    url = 'http://192.168.0.195:5000/generate'
    payload = {'firebase_uid': firebase_uid}
    headers = {'Content-Type': 'application/json'}
    response = requests.post(url, json=payload, headers=headers)
    return response
