#!/bin/bash
set -e

echo "Preparing for Render.com deployment..."

# Create necessary directories
echo "Creating required directories..."
mkdir -p rag_data/vector_store
mkdir -p templates
mkdir -p static

# Check if RAG data exists
if [ ! -f "rag_data/prof_summary.txt" ]; then
    echo "Warning: rag_data/prof_summary.txt not found. Creating a placeholder..."
    echo "Professor Risa Wechsler is an astrophysicist and cosmologist specializing in dark matter studies and large-scale structure of the universe." > rag_data/prof_summary.txt
fi

echo "Deployment preparation complete!"
echo ""
echo "To deploy to Render.com:"
echo "1. Push this code to a GitHub repository"
echo "2. Visit https://dashboard.render.com/"
echo "3. Create a new Web Service from your GitHub repository"
echo "4. Use the following settings:"
echo "   - Build Command: pip install -r requirements.txt"
echo "   - Start Command: uvicorn app:app --host 0.0.0.0 --port \$PORT"
echo ""
echo "For more detailed instructions, see RENDER_DEPLOY.md" 