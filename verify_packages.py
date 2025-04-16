#!/usr/bin/env python
"""
Verify package installation for Render deployment.
This script checks if critical packages are installed correctly.
"""
import sys
import pkg_resources

print("Python version:", sys.version)
print("\nChecking required packages...")

REQUIRED_PACKAGES = [
    "google-generativeai",
    "langchain-google-genai",
    "fastapi",
    "uvicorn",
    "langchain",
    "faiss-cpu"
]

for package in REQUIRED_PACKAGES:
    try:
        version = pkg_resources.get_distribution(package).version
        print(f"✅ {package}=={version}")
    except pkg_resources.DistributionNotFound:
        print(f"❌ {package} is NOT installed!")
        if package == "google-generativeai":
            print("Attempting to install google-generativeai...")
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", "google-generativeai==0.3.2"])
            try:
                version = pkg_resources.get_distribution(package).version
                print(f"✅ Now installed: {package}=={version}")
            except:
                print("❌ Still failed to install google-generativeai")

# Try to import google.generativeai directly
print("\nTrying to import google.generativeai...")
try:
    import google.generativeai
    print("✅ Successfully imported google.generativeai")
except ImportError as e:
    print(f"❌ Failed to import: {e}")

print("\nPackage verification complete.") 