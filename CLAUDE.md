# SHIRT//AI — Spreadshirt Design Agent

## What It Does

An AI-powered Flask web app that automatically generates t-shirt designs and uploads them to Spreadshirt. Given a niche (e.g. "coffee lovers", "gym rats"), the agent:

1. Uses **Claude** (claude-opus-4-6) to brainstorm design concepts and slogans
2. Generates designs in one of two modes:
   - **SVG mode** (default): Claude generates vector SVG artwork directly
   - **DALL·E mode**: OpenAI DALL·E 3 generates a raster image, then background is auto-removed
3. Uses Claude to write SEO-optimized titles, descriptions, and tags for each design
4. Optionally shows a **human review gate** before uploading — approve, skip, or request a regeneration with feedback
5. Uploads the finished design + product listing to the Spreadshirt Partner API
   - If no Spreadshirt credentials are provided, runs in **dry-run mode** (generates designs but doesn't upload)

Streaming progress is shown in real-time in the browser UI via Server-Sent Events (SSE).

---

## File Structure

```
spreadshirt-agent/
├── app.py              ← Flask backend + agent logic (all-in-one)
│   ├── Routes:         /  /defaults  /start  /stop/<id>  /stream/<id>
│   │                   /review/<id>  /review_data/<id>
│   ├── run_agent()     ← Main agent loop (runs in background thread)
│   ├── generate_concepts()  ← Claude brainstorms slogans/concepts
│   ├── generate_svg()       ← Claude generates SVG code
│   ├── generate_dalle_image() ← DALL·E 3 raster generation
│   ├── generate_metadata()  ← Claude writes SEO metadata
│   └── upload_to_spreadshirt() ← Spreadshirt Partner API upload
├── templates/
│   └── index.html      ← Single-page frontend UI
├── .env.example        ← Template for environment variables
├── .env                ← Local env file with actual keys (not committed to git)
└── README.md
```

---

## How to Run

### 1. Install dependencies
```bash
pip install flask flask-cors anthropic openai requests pillow rembg
```

Optional (recommended):
```bash
pip install cairosvg   # SVG → PNG conversion (without this, SVGs are uploaded as-is)
pip install python-dotenv  # auto-loads .env file
```

### 2. Set up environment variables
Fill in your keys in `.env` (already in place). `python-dotenv` will load it automatically on startup.

### 3. Start the server
```bash
python app.py
```

### 4. Open in browser
```
http://localhost:5000
```

---

## API Keys & Credentials (from `.env` file)

| Variable | Value | Source |
|---|---|---|
| `ANTHROPIC_KEY` | *(set your own)* | [console.anthropic.com](https://console.anthropic.com) |
| `OPENAI_KEY` | *(set your own)* | [platform.openai.com](https://platform.openai.com) |
| `SPREADSHIRT_KEY` | *(set your own)* | Spreadshirt Partner Area → API |
| `SPREADSHIRT_USER` | `115239581` | Your numeric Spreadshirt user ID |

> **Warning:** The `env` file currently has the same placeholder value for `OPENAI_KEY` and `SPREADSHIRT_KEY` as the Anthropic key — these need to be replaced with the actual keys for those services.

---

## Key Behaviors & Config

- **Product type**: Default is `210` (Unisex Crew Neck). Change `productTypeId` in `upload_to_spreadshirt()` in `app.py`.
- **Review mode**: Enabled by default. Each design pauses for human approval before upload (10-minute timeout).
- **Inspiration image**: Users can upload a reference image; Claude will match its aesthetic style.
- **Dry-run mode**: Automatically activated when `SPREADSHIRT_KEY` or `SPREADSHIRT_USER` is missing.
- **SVG conversion**: If `cairosvg` is installed, SVGs are converted to PNG before upload. Otherwise SVG is uploaded directly (Spreadshirt accepts both).
- **Background removal**: Uses `rembg` if available; falls back to simple white-pixel removal.
- **Claude model**: `claude-opus-4-6` (used for concepts, SVG generation, and metadata).
