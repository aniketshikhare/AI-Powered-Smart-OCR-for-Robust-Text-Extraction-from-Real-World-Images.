"""
Test the API endpoints
"""

import requests
import json

def test_health():
    """Test health endpoint"""
    try:
        response = requests.get('http://127.0.0.1:5000/api/health')
        print("Health Check:")
        print(json.dumps(response.json(), indent=2))
        print()
    except Exception as e:
        print(f"Health check failed: {e}")

def test_languages():
    """Test languages endpoint"""
    try:
        response = requests.get('http://127.0.0.1:5000/api/languages')
        print("Supported Languages:")
        print(json.dumps(response.json(), indent=2))
        print()
    except Exception as e:
        print(f"Languages endpoint failed: {e}")

def test_extract():
    """Test text extraction endpoint"""
    try:
        image_path = r'C:\Users\dell\source\repos\AI-Powered-Smart-OCR-for-Robust-Text-Extraction-from-Real-World-Images\data\images\test1.jpg'
        with open(image_path, 'rb') as f:
            files = {'file': f}
            data = {'languages': 'en', 'use_gpu': 'false'}
            response = requests.post('http://127.0.0.1:5000/api/extract', files=files, data=data)
        
        print("Text Extraction Result:")
        print(json.dumps(response.json(), indent=2))
        print()
    except Exception as e:
        print(f"Text extraction failed: {e}")

if __name__ == "__main__":
    print("Testing API Endpoints...\n")
    test_health()
    test_languages()
    test_extract()