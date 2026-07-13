# Deploy this app as a public website
====================================

The TEM Particle Analyzer is a Streamlit app (`app/streamlit_app.py`).
Anyone with the public URL (or a QR code pointing at it) can:

1. Choose **Rods** or **Dots**
2. Enter the **scale bar length in nm** (pixels are measured automatically)
3. Upload a TEM image
4. Approve/discard outlines and download a CSV

## Your Cloud deploy

After fixing deps, reopen or reboot the app at Streamlit Cloud. Expected URL:

```text
https://tem-analysis-y7v8uc3xf2fxfayixohzen.streamlit.app/
```

(Or whatever subdomain your Streamlit dashboard shows.)

## Recommended settings (Streamlit Community Cloud)

1. Open https://share.streamlit.io and sign in with GitHub.
2. App settings:
   - **Repository:** `fayh601-glitch/TEM-analysis`
   - **Branch:** `main`
   - **Main file path:** `app/streamlit_app.py`
3. Ensure `runtime.txt` exists in the repo (`python-3.11`) — Cloud must not use Python 3.14.
4. Click **Reboot app** / **Redeploy** after pulling the latest `main`.

### Why the first deploy failed

Cloud defaulted to **Python 3.14**, and pinned packages like `numpy==1.26.4` /
`Pillow==10.4.0` / `scipy==1.11.4` have no wheels there, so installs tried to
compile from source and crashed. The repo now pins **Python 3.11** and uses
version ranges with binary wheels.

### Make a QR code

```bash
pip install 'qrcode[pil]'
python scripts/make_app_qr.py "https://tem-analysis-y7v8uc3xf2fxfayixohzen.streamlit.app"
```

## Local testing

```bash
cd TEM-analysis
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .
streamlit run app/streamlit_app.py
```

Open http://localhost:8501

## GitHub

https://github.com/fayh601-glitch/TEM-analysis
