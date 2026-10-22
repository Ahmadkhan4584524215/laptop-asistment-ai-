# 💻 LapScope – AI Laptop Analyzer

Type a laptop's full name and get:

- **Core hardware**: CPU, GPU, RAM, storage type and size, battery, display resolution, refresh rate, panel type, ports
- **Performance scores** for **gaming** (with estimated FPS) and **workstation** use
- **Side-by-side comparison** of 2–4 laptops, with charts and an AI verdict
- **Prices and value-for-money**, looked up with Google Search grounding, in your local currency

Built with [Streamlit](https://streamlit.io) and Google Gemini Flash (`google-genai` SDK).

> ⚠️ Specs, scores and prices are AI-generated estimates. Laptops ship in many configurations,
> so always verify on the manufacturer's page and the seller's listing before buying.

---

## 1. Run locally

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add your free Gemini key (get one at https://aistudio.google.com/apikey)
export GEMINI_API_KEY="your-key"  # Windows PowerShell: $env:GEMINI_API_KEY="your-key"

# 4. Start the app
streamlit run app.py
```

You can also skip step 3 and paste the key into the app's sidebar.

### Configuration

| Setting | Where | Default |
|---|---|---|
| `GEMINI_API_KEY` | env var or Streamlit secrets | none (paste in sidebar) |
| `GEMINI_MODEL` | env var, secret, or sidebar | `gemini-3.8-flash` |
| Price region/currency | sidebar | Pakistan (PKR) |

If the model name is retired or unavailable, the app automatically tries `gemini-3.5-flash`
and `gemini-flash-latest`, then asks the API which Flash models your key can use and picks the
newest one. You can also change the model in the sidebar or via `GEMINI_MODEL`.

---

## 2. Push to GitHub

1. Create a `.gitignore` file so secrets never get uploaded:

   ```
   .venv/
   __pycache__/
   .env
   .streamlit/secrets.toml
   ```

2. Create an **empty** repository on GitHub (no README) named e.g. `lapscope`.
3. In your project folder (containing `app.py`, `requirements.txt`, `README.md`, `.gitignore`):

   ```bash
   git init
   git add .
   git commit -m "Initial commit: LapScope"
   git branch -M main
   git remote add origin https://github.com/<your-username>/lapscope.git
   git push -u origin main
   ```

   GitHub no longer accepts account passwords for `git push`. When prompted, use a
   **Personal Access Token** (GitHub → Settings → Developer settings → Personal access tokens),
   or sign in with GitHub CLI (`gh auth login`).

**Never commit your API key.**

---

## 3. Deploy on Streamlit Community Cloud (free)

1. Go to <https://share.streamlit.io> and sign in with GitHub.
2. Click **Create app** → **Deploy a public app from GitHub**.
3. Choose your repository, branch `main`, and main file path `app.py`.
4. Open **Advanced settings → Secrets** and paste (optional, see below):

   ```toml
   GEMINI_API_KEY = "your-key"
   ```

5. Click **Deploy**. After a minute or two you get a public `*.streamlit.app` URL.

Every `git push` to `main` redeploys the app automatically.

### Whose API key should the public app use?

- **No secret set (default):** each visitor pastes their own free Gemini key. Your quota is safe.
- **Secret set:** all visitors use *your* key and your quota. Fine for demos. For a public app,
  keep your Google AI Studio quota/billing limits low, or make the app private under
  *Settings → Sharing*.

---

## How it works

| Feature | Approach |
|---|---|
| Specs and performance | Gemini returns strict JSON, which is validated and cleaned (scores clamped to 0–100, bad fields replaced by "Unknown") |
| Prices | Gemini with Google Search grounding. If search is unavailable it falls back to a clearly labelled estimate |
| Value index | Performance score ÷ typical price, scaled so the best laptop = 100 |
| Speed and quota | Laptops are fetched in parallel, results are cached per session, and rate limits are retried automatically |

The free Gemini tier has rate limits. Comparing 4 laptops with prices makes about 9 requests,
so if you see a "busy" message wait a minute and press the button again (cached results are reused).

## Project files

```
app.py            # the whole Streamlit app
requirements.txt  # Python dependencies
README.md         # this file
```

## Ideas to extend

- Upload a photo of the laptop's box/sticker and let Gemini read the model name
- Save favourite laptops to a database
- Add benchmark data (Geekbench, 3DMark) for more grounded scores
