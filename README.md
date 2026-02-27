# SHIRT//AI — Spreadshirt Design Agent

## Setup (2 minutes)

### 1. Install Python dependencies
```bash
pip install flask flask-cors anthropic openai requests pillow rembg
```

### 2. Run the app
```bash
python app.py
```

### 3. Open in browser
```
http://localhost:5000
```

---

## What you need

| Key | Where to get it |
|-----|----------------|
| **Anthropic API Key** | console.anthropic.com |
| **OpenAI API Key** | platform.openai.com |
| **Spreadshirt API Key** | spreadshirt.com → Partner Area → API |
| **Spreadshirt User ID** | Your numeric user ID from your account URL |

Leave the Spreadshirt fields blank to run in **dry-run mode** — it will generate all designs and show them in the UI without uploading.

---

## Product Type IDs (Spreadshirt)

The default is `210` (Unisex Crew Neck). To change the shirt style, edit `productTypeId` in `app.py`. Common IDs:
- `210` — Unisex Crew Neck
- `812` — Unisex Heavy Cotton Tee
- `175` — Women's V-Neck

You can find your region's IDs at: `https://api.spreadshirt.net/api/v1/productTypes`

---

## Tips for best results

- **Image upscaling**: DALL·E generates at 1024px. Spreadshirt recommends 4500px. Install `realesrgan` for automatic upscaling.
- **Background removal**: `rembg` is used automatically if installed. For better results use the remove.bg API (add your key in `app.py`).
- **Rate limits**: The agent adds a small delay between designs automatically. For large batches (20+), expect ~3 min per design.

---

## File structure
```
spreadshirt-agent/
├── app.py              ← Flask backend + agent logic
├── templates/
│   └── index.html      ← Frontend UI
└── README.md
```
