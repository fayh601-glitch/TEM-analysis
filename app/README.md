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

## Output

Download the overlay PNG and measurements CSV from the page. Results are also written to a temporary folder during the session.
