#!/usr/bin/env python3
"""Simple test script to verify the upload endpoint is working."""

import requests
import sys

API_BASE_URL = "http://localhost:8000"

# Test health endpoint
print("Testing API health...")
try:
    response = requests.get(f"{API_BASE_URL}/health", timeout=5)
    if response.status_code == 200:
        print("✓ API is running")
        print(f"  Response: {response.json()}")
    else:
        print(f"✗ API returned status {response.status_code}")
        sys.exit(1)
except Exception as e:
    print(f"✗ Cannot connect to API: {e}")
    print(f"  Make sure the server is running at {API_BASE_URL}")
    sys.exit(1)

# Test with a dummy file (this will fail but show the error)
print("\nTesting upload endpoint (this will show any configuration errors)...")
try:
    # Create a dummy PDF-like file
    dummy_content = b"%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n>>\nendobj\nxref\n0 0\ntrailer\n<<\n/Root 1 0 R\n>>\nstartxref\n9\n%%EOF"
    
    files = {"file": ("test.pdf", dummy_content, "application/pdf")}
    response = requests.post(
        f"{API_BASE_URL}/upload",
        files=files,
        timeout=30
    )
    
    print(f"  Status code: {response.status_code}")
    if response.status_code == 201:
        print("✓ Upload endpoint is working!")
        print(f"  Response: {response.json()}")
    else:
        print(f"✗ Upload failed with status {response.status_code}")
        try:
            error_data = response.json()
            print(f"  Error: {error_data}")
        except:
            print(f"  Error text: {response.text[:500]}")
except Exception as e:
    print(f"✗ Error testing upload: {e}")

print("\nDone!")

