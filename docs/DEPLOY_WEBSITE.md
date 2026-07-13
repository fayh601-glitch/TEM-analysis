# Deploy this app as a public website
====================================

The TEM Particle Analyzer is a Streamlit app (`app/streamlit_app.py`).
Anyone with the public URL (or a QR code pointing at it) can:

1. Choose **Rods** or **Dots**
2. Enter **scale bar (nm)** and **scale bar (pixels)**
3. Upload a TEM image
4. Approve/discard outlines and download a CSV

## Recommended host: Streamlit Community Cloud (free)

1. Push this repo to GitHub (already at
   https://github.com/fayh601-glitch/TEM-analysis).
2. Open https://share.streamlit.io and sign in with GitHub.
3. Click **New app** and set:
   - **Repository:** `fayh601-glitch/TEM-analysis`
   - **Branch:** `main`
   - **Main file path:** `app/streamlit_app.py`
   - **Python packages file:** `requirements-web.txt`
     (Advanced settings → if available)
4. Click **Deploy**.

Your public link will look like:

```text
https://tem-analysis-<username>.streamlit.app
```

or a custom subdomain you choose at deploy time.

### Make a QR code for posters / lab notebook

Once you have the public URL:

```bash
# from TEM-analysis with venv active
pip install qrcode[pil]
python scripts/make_app_qr.py "https://YOUR-APP.streamlit.app"
```

This writes `docs/tem_analyzer_qr.png` and prints the link.

## Local testing (this machine only)

```bash
cd TEM-analysis
source .venv/bin/activate
pip install -r requirements-web.txt
streamlit run app/streamlit_app.py
```

Open http://localhost:8501 — this is **not** a public website until Cloud is deployed.

## GitHub link (source code)

https://github.com/fayh601-glitch/TEM-analysis
