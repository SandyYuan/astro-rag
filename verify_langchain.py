#!/usr/bin/env python
"""
Verify LangChain module imports for debugging the Render deployment.
This script checks specific LangChain import paths that are causing issues.
"""
import sys
import os
import traceback
import importlib

print("=" * 50)
print("LANGCHAIN VERIFICATION SCRIPT")
print("Python version:", sys.version)
print("Python executable:", sys.executable)
print("-" * 50)

# Check langchain version
try:
    import langchain
    print(f"LangChain version: {getattr(langchain, '__version__', 'unknown')}")
    print(f"LangChain module path: {langchain.__file__}")
except Exception as e:
    print("Failed to import langchain:", str(e))
    traceback.print_exc()

# Check vectorstores module path
print("\nChecking vectorstores imports:")
try:
    # Check if vectorstores exists under langchain
    import langchain.vectorstores
    print("✅ langchain.vectorstores exists")
    print(f"Module path: {langchain.vectorstores.__file__}")
    
    # Check if FAISS is available
    try:
        from langchain.vectorstores import FAISS
        print("✅ langchain.vectorstores.FAISS exists")
    except ImportError:
        print("❌ langchain.vectorstores.FAISS not found")
        
except ImportError:
    print("❌ langchain.vectorstores module not found")
    
    # Check if it's under langchain_community instead
    try:
        import langchain_community.vectorstores
        print("⚠️ vectorstores exists under langchain_community instead")
        print(f"Module path: {langchain_community.vectorstores.__file__}")
        
        # Check if LangChain Community is installed
        import langchain_community
        print(f"LangChain Community version: {getattr(langchain_community, '__version__', 'unknown')}")
    except ImportError:
        print("❌ langchain_community.vectorstores not found either")

# Check document loaders
print("\nChecking document loaders:")
try:
    import langchain.document_loaders
    print("✅ langchain.document_loaders exists")
    
    try:
        from langchain.document_loaders import PyPDFLoader
        print("✅ langchain.document_loaders.PyPDFLoader exists")
    except ImportError:
        print("❌ PyPDFLoader not found in langchain.document_loaders")
except ImportError:
    print("❌ langchain.document_loaders module not found")
    
    # Check if it's under langchain_community instead
    try:
        import langchain_community.document_loaders
        print("⚠️ document_loaders exists under langchain_community instead")
    except ImportError:
        print("❌ document_loaders not found in langchain_community either")

# Check package installation for both versions
print("\nChecking package installations:")
packages_to_check = ["langchain", "langchain-community"]
for package in packages_to_check:
    try:
        spec = importlib.util.find_spec(package.replace("-", "_"))
        if spec:
            print(f"✅ {package} found at: {spec.origin}")
        else:
            try:
                import pkg_resources
                version = pkg_resources.get_distribution(package).version
                print(f"✅ {package}=={version} (installed via pkg_resources)")
            except pkg_resources.DistributionNotFound:
                print(f"❌ {package} not installed according to pkg_resources")
    except Exception as e:
        print(f"❌ Error checking {package}: {str(e)}")

print("=" * 50) 