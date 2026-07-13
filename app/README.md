# Web upload app

Browser-based interface for analyzing TEM images without memorizing Terminal flags.

## Run

```bash
cd TEM-analysis
source .venv/bin/activate
pip install -r requirements-optional.txt
streamlit run app/streamlit_app.py
```

Your browser opens automatically. Upload an image, set the scale bar, choose **nanorods only** for Enright-style samples, and click **Analyze**.

## Human review

After analysis:

1. Green numbered markers = **keep**, red = **discard**.
2. Click a marker on the interactive overlay to toggle keep/discard.
3. Or use the particle table checkboxes below the plot.
4. Download **approved measurements CSV** — only kept particles are included.

## Output

Download the approved CSV and original overlay PNG from the page. Session files are also written under `outputs/streamlit_sessions/`.
