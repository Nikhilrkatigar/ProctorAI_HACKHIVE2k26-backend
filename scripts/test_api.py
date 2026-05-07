import requests
import sys
import json

API = "http://127.0.0.1:8000"

def test_api():
    print("Testing Backend API...")
    
    # Test connection
    try:
        res = requests.get(f"{API}/")
        print(f"Index: {res.status_code}")
    except Exception as e:
        print(f"Failed to connect to API: {e}")
        return

    # Try signing up a student
    student_data = {
        "candidate_id": "test_student_1",
        "name": "Test Student 1",
        "email": "student1@test.com",
        "password": "password123",
        "role": "student"
    }
    
    res = requests.post(f"{API}/auth/signup", json=student_data)
    print(f"Student Signup: {res.status_code} {res.text}")
    
    # Login student
    res = requests.post(f"{API}/auth/login", data={"username": "test_student_1", "password": "password123"})
    print(f"Student Login: {res.status_code} {res.text}")
    
    if res.status_code == 200:
        student_token = res.json().get("access_token")
    else:
        student_token = None

    # Try signing up a proctor
    proctor_data = {
        "candidate_id": "test_proctor_1",
        "name": "Test Proctor 1",
        "email": "proctor1@test.com",
        "password": "password123",
        "role": "proctor"
    }
    
    res = requests.post(f"{API}/auth/signup", json=proctor_data)
    print(f"Proctor Signup: {res.status_code} {res.text}")
    
    # Login proctor
    res = requests.post(f"{API}/auth/login", data={"username": "test_proctor_1", "password": "password123"})
    print(f"Proctor Login: {res.status_code} {res.text}")

    if res.status_code == 200:
        proctor_token = res.json().get("access_token")
    else:
        proctor_token = None

if __name__ == '__main__':
    test_api()
