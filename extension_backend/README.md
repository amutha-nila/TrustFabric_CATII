# Optional browser-extension service

This folder contains the original TrustFabric extension backend for the separate Chrome extension in `../extension/`. It runs on port 8765 and is separate from the CAT II ML classifier API in `../app/` (port 8000).

Follow the project-root README only for the assessed CAT II ML pipeline. From the project root, start this optional backend with the existing virtual environment:

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --app-dir extension_backend --host 127.0.0.1 --port 8765
```

The extension’s claim-verification API remains at `http://127.0.0.1:8000`; privacy reports and media analysis use this separate backend at `http://127.0.0.1:8765`. The media endpoint may return a model-loading error if its configured classifier is unavailable; the privacy endpoint returns only maintained policy-navigation summaries. This backend is not used to produce the CAT II metrics.
