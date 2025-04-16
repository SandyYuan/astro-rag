#!/usr/bin/env python
"""
Simple script to test importing google.generativeai
This will be called during the Render build process
"""
import sys
import os
import traceback

print("=" * 50)
print("IMPORT TESTER SCRIPT")
print("Python version:", sys.version)
print("Python executable:", sys.executable)
print("Current working directory:", os.getcwd())
print("-" * 50)

# Try standard approach
print("\nApproach 1: import google.generativeai")
try:
    import google.generativeai
    print("✅ SUCCESS! google.generativeai imported successfully")
    print(f"Module path: {google.generativeai.__file__}")
    print(f"Version: {getattr(google.generativeai, '__version__', 'unknown')}")
except Exception as e:
    print("❌ FAILED!")
    print(f"Error: {str(e)}")
    traceback.print_exc()

# Try alternate approach
print("\nApproach 2: from google import generativeai")
try:
    from google import generativeai
    print("✅ SUCCESS! from google import generativeai worked")
    print(f"Module path: {generativeai.__file__}")
    print(f"Version: {getattr(generativeai, '__version__', 'unknown')}")
except Exception as e:
    print("❌ FAILED!")
    print(f"Error: {str(e)}")
    traceback.print_exc()

# Check module path
print("\nChecking sys.path:")
for i, path in enumerate(sys.path):
    print(f"{i}: {path}")

# Check if google package exists
print("\nLooking for google package:")
google_paths = []
for path in sys.path:
    potential_google = os.path.join(path, "google")
    if os.path.exists(potential_google):
        google_paths.append(potential_google)
        print(f"Found at: {potential_google}")
        # List contents
        if os.path.isdir(potential_google):
            print("  Contents:", os.listdir(potential_google))

print("=" * 50) 