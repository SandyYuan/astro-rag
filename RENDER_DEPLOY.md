# Deploying Astro-RAG to Render.com

This guide provides instructions for deploying the Astro-RAG application to Render.com.

## Prerequisites

1. A Render.com account (free tier is sufficient)
2. Your code pushed to a GitHub repository

## Deployment Steps

### Option 1: Deploy from GitHub (Recommended)

1. Log in to your Render.com account
2. Click "New +" and select "Web Service"
3. Connect your GitHub account if you haven't already
4. Select the repository containing your code
5. Configure the service:
   - **Name**: Choose a name (e.g., "astro-rag")
   - **Environment**: Select "Python 3"
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn app:app --host 0.0.0.0 --port $PORT`
6. Select the free instance type
7. Click "Create Web Service"

### Option 2: Deploy using render.yaml

1. Ensure the `render.yaml` file is in your repository
2. Go to your Render.com dashboard
3. Click "Blueprint" from the navigation menu
4. Connect to your GitHub repository
5. Render will automatically detect the render.yaml file and create the service

## After Deployment

1. Once deployed, Render will provide a public URL for your application (e.g., `https://astro-rag.onrender.com`)
2. Visit this URL in your browser
3. Enter your Google Gemini API key when prompted
4. Start chatting with the astronomy chatbot!

## Notes

- The free tier of Render.com may spin down your service after periods of inactivity, which can lead to a short delay when it's accessed again
- You can upgrade to a paid plan for better performance and to avoid spin-downs
- Remember that users will need to provide their own Gemini API keys 