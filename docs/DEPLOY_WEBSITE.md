# Deploy this app as a public website
====================================

## Critical: Python version (read this)

Streamlit Community Cloud **ignores** `runtime.txt`. Your last log showed:

```text
Using Python 3.14.6 environment
```

That is why installs hang or fail. You must pick the Python version in the
**Streamlit website UI**, not in GitHub files.

### Fix (delete + redeploy)

1. Go to https://share.streamlit.io
2. Open your app → **⋮** → **Delete**
3. Click **Create app** / **New app**
4. Fill in:
   - Repository: `fayh601-glitch/TEM-analysis`
   - Branch: `main`
   - Main file: `app/streamlit_app.py`
5. Open **Advanced settings**
6. Set **Python version** to **3.11** or **3.12** (not 3.13 / 3.14)
7. Deploy and wait (first install of scipy/skimage can take 5–15 minutes)

After it works, your public URL will be something like:

```text
https://tem-analysis-y7v8uc3xf2fxfayixohzen.streamlit.app/
```

### Check logs while it cooks

**Manage app → Logs**

You want to see:

```text
Using Python 3.11...   (or 3.12)
```

If it still says `Python 3.14`, delete again and make sure Advanced settings
were set **before** clicking Deploy.

## What the app needs from users

1. Choose **Rods** or **Dots**
2. Enter scale bar **nm** (pixels are auto-measured)
3. Upload TEM image
4. Approve/discard outlines → download CSV

## Local test

```bash
cd TEM-analysis
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .
streamlit run app/streamlit_app.py
```

## GitHub

https://github.com/fayh601-glitch/TEM-analysis
