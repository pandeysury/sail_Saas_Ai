#!/usr/bin/env python3
# test_dashboard.py - Test dashboard API directly

import requests
import json
from requests.auth import HTTPBasicAuth

BASE_URL = "http://localhost:8000"
AUTH = HTTPBasicAuth("admin", "admin123")

def test_endpoints():
    print("🧪 Testing Dashboard API...")
    
    # Test 1: Basic API
    try:
        response = requests.get(f"{BASE_URL}/api/dashboard/test")
        print(f"✅ Basic API: {response.status_code} - {response.json()}")
    except Exception as e:
        print(f"❌ Basic API failed: {e}")
        return
    
    # Test 2: Auth API
    try:
        response = requests.get(f"{BASE_URL}/api/dashboard/test-auth", auth=AUTH)
        print(f"✅ Auth API: {response.status_code} - {response.json()}")
    except Exception as e:
        print(f"❌ Auth API failed: {e}")
        return
    
    # Test 3: Overview
    try:
        response = requests.get(f"{BASE_URL}/api/dashboard/overview", auth=AUTH)
        data = response.json()
        print(f"✅ Overview: {response.status_code}")
        print(f"   Total queries: {data.get('total_queries', 'N/A')}")
        print(f"   Avg quality: {data.get('avg_quality', 'N/A')}")
    except Exception as e:
        print(f"❌ Overview failed: {e}")
    
    # Test 4: Queries
    try:
        response = requests.get(f"{BASE_URL}/api/dashboard/queries?limit=5", auth=AUTH)
        data = response.json()
        print(f"✅ Queries: {response.status_code}")
        print(f"   Found {len(data.get('queries', []))} queries")
    except Exception as e:
        print(f"❌ Queries failed: {e}")

if __name__ == "__main__":
    test_endpoints()