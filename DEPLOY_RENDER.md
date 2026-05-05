# Deploy Pravesh Saathi on Render

## 1) Push code to GitHub

Make sure this repository is pushed to GitHub and includes:
- `app.py`
- `requirements.txt`
- `render.yaml`
- `vector_db/` (required for retrieval)

## 2) Create service with Blueprint

1. Open [https://dashboard.render.com](https://dashboard.render.com)
2. Click **New +** -> **Blueprint**
3. Connect/select your GitHub repo
4. Render will read `render.yaml` and create service `pravesh-saathi-ai`

## 3) Set required environment variable

In Render dashboard, open the created service and set:
- `GROQ_API_KEY` = your Groq API key

Optional:
- `WEBSITE_REFRESH_MINUTES` (default `180`) to control automatic UIET website context refresh.

## 4) Deploy and verify

After deploy finishes:
- Open the Render URL
- Health check endpoint: `/`
- Chat API endpoint: `/chat`

## Notes

- This app uses `torch` + `sentence-transformers`, so startup can be slow on free plans.
- If free plan fails due to memory/cold starts, use a paid instance.
- The app now auto-refreshes website context in background at runtime, so manual rescrape is usually not needed.
