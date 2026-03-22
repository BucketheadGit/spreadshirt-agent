"""
Spreadshirt AI Design Agent — Backend
Run with: python app.py  |  Open http://localhost:5000
"""

import io, json, base64, time, threading, uuid, re, os, datetime
from flask import Flask, render_template, request, jsonify, Response, stream_with_context
from flask_cors import CORS
import anthropic
import openai
import requests
from PIL import Image

# Load .env if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    from rembg import remove as rembg_remove
    REMBG_AVAILABLE = True
except ImportError:
    REMBG_AVAILABLE = False

# Try cairosvg for SVG→PNG conversion
try:
    import cairosvg
    CAIROSVG_AVAILABLE = True
except (ImportError, OSError):
    CAIROSVG_AVAILABLE = False

app = Flask(__name__)
CORS(app)

# ─── Job store ────────────────────────────────────────────────────────────────
jobs = {}
SPREADSHIRT_BASE = "https://api.spreadshirt.net/api/v1"
OUTPUTS_DIR = os.path.join(os.path.dirname(__file__), "outputs")
HISTORY_FILE = os.path.join(os.path.dirname(__file__), "upload_history.json")
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# ─── Autopilot state ──────────────────────────────────────────────────────────
autopilot_state = {
    "enabled": False,
    "interval_hours": 24,
    "designs_per_run": 3,
    "last_run": None,
    "next_run": None,
    "runs": [],          # list of {timestamp, niche, job_id, uploads}
    "thread": None,
}
_autopilot_lock = threading.Lock()


# ─── Upload history helpers ───────────────────────────────────────────────────

def load_history():
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return {"uploads": [], "niches_used": []}


def save_history(history):
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)


def record_upload(niche, slogan, title, product_url=None):
    history = load_history()
    history["uploads"].append({
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "niche": niche,
        "slogan": slogan,
        "title": title,
        "product_url": product_url,
    })
    if niche not in history["niches_used"]:
        history["niches_used"].append(niche)
    save_history(history)

# ─── Routes ──────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/defaults")
def defaults():
    """Return env-file defaults so the UI can pre-fill keys."""
    return jsonify({
        "anthropic_key": os.environ.get("ANTHROPIC_KEY", ""),
        "openai_key":    os.environ.get("OPENAI_KEY", ""),
        "ss_key":        os.environ.get("SPREADSHIRT_KEY", ""),
        "ss_user":       os.environ.get("SPREADSHIRT_USER", ""),
    })


@app.route("/start", methods=["POST"])
def start_job():
    data = request.json
    job_id = str(uuid.uuid4())[:8]
    jobs[job_id] = {
        "status": "running", "events": [], "results": [],
        "cancelled": False, "review_pending": None
    }
    threading.Thread(target=run_agent, args=(job_id, data), daemon=True).start()
    return jsonify({"job_id": job_id})


@app.route("/stop/<job_id>", methods=["POST"])
def stop_job(job_id):
    job = jobs.get(job_id)
    if not job:
        return jsonify({"ok": False}), 404
    job["cancelled"] = True
    emit(job_id, "warn", "🛑 Stop requested…")
    if job["review_pending"]:
        job["review_pending"]["decision"] = {"action": "skip"}
        job["review_pending"]["event"].set()
    return jsonify({"ok": True})


@app.route("/review_data/<job_id>")
def review_data(job_id):
    """Frontend fetches this separately to avoid SSE payload limits."""
    job = jobs.get(job_id)
    if not job or not job["review_pending"]:
        return jsonify({"ok": False}), 404
    rp = job["review_pending"]
    return jsonify({
        "ok": True,
        "image_b64": rp["image_b64"],
        "metadata":  rp["metadata"],
        "slogan":    rp["slogan"],
        "index":     rp["index"],
        "file_type": rp.get("file_type", "png"),
    })


@app.route("/review/<job_id>", methods=["POST"])
def review_decision(job_id):
    job = jobs.get(job_id)
    if not job or not job["review_pending"]:
        return jsonify({"ok": False, "reason": "No review pending"}), 400
    job["review_pending"]["decision"] = request.json
    job["review_pending"]["event"].set()
    return jsonify({"ok": True})


@app.route("/history")
def get_history():
    return jsonify(load_history())


@app.route("/suggest_niches", methods=["POST"])
def suggest_niches():
    """Ask Claude to suggest trending, profitable niches, avoiding already-used ones."""
    data = request.json or {}
    ant_key = data.get("anthropic_key") or os.environ.get("ANTHROPIC_KEY", "")
    if not ant_key:
        return jsonify({"ok": False, "error": "No Anthropic key"}), 400
    history = load_history()
    used = history.get("niches_used", [])
    try:
        ant = anthropic.Anthropic(api_key=ant_key)
        niches = discover_niches(ant, used, count=10)
        return jsonify({"ok": True, "niches": niches})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/autopilot", methods=["GET"])
def autopilot_status():
    with _autopilot_lock:
        return jsonify({
            "enabled": autopilot_state["enabled"],
            "interval_hours": autopilot_state["interval_hours"],
            "designs_per_run": autopilot_state["designs_per_run"],
            "last_run": autopilot_state["last_run"],
            "next_run": autopilot_state["next_run"],
            "runs": autopilot_state["runs"][-20:],  # last 20
        })


@app.route("/autopilot", methods=["POST"])
def autopilot_configure():
    data = request.json or {}
    with _autopilot_lock:
        if "enabled" in data:
            autopilot_state["enabled"] = bool(data["enabled"])
        if "interval_hours" in data:
            autopilot_state["interval_hours"] = max(1, int(data["interval_hours"]))
        if "designs_per_run" in data:
            autopilot_state["designs_per_run"] = max(1, min(20, int(data["designs_per_run"])))
        # Store credentials for unattended runs
        for k in ("anthropic_key", "openai_key", "spreadshirt_key", "spreadshirt_user",
                  "design_mode", "style"):
            if k in data:
                autopilot_state[k] = data[k]
        if autopilot_state["enabled"]:
            _schedule_next_run()
    return jsonify({"ok": True, **{k: autopilot_state[k]
                                    for k in ("enabled", "interval_hours", "designs_per_run",
                                              "last_run", "next_run")}})


@app.route("/stream/<job_id>")
def stream(job_id):
    def generate():
        sent = 0
        while True:
            job = jobs.get(job_id)
            if not job:
                yield f"data: {json.dumps({'type':'error','msg':'Job not found'})}\n\n"
                return
            while sent < len(job["events"]):
                yield f"data: {json.dumps(job['events'][sent])}\n\n"
                sent += 1
            if job["status"] in ("done", "error", "cancelled"):
                yield f"data: {json.dumps({'type':'done','results':job['results'],'status':job['status']})}\n\n"
                return
            time.sleep(0.3)
    return Response(stream_with_context(generate()), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ─── Agent ────────────────────────────────────────────────────────────────────

def emit(job_id, type_, msg, **kwargs):
    jobs[job_id]["events"].append({"type": type_, "msg": msg, **kwargs})


def is_cancelled(job_id):
    return jobs.get(job_id, {}).get("cancelled", False)


def run_agent(job_id, data):
    try:
        niche         = data["niche"]
        count         = int(data.get("count", 5))
        ant_key       = data["anthropic_key"]
        oai_key       = data.get("openai_key", "").strip()
        ss_key        = data.get("spreadshirt_key", "").strip()
        ss_user       = data.get("spreadshirt_user", "").strip()
        dry_run       = not (ss_key and ss_user)
        design_mode   = data.get("design_mode", "svg")   # "svg" or "dalle"
        img_style     = data.get("style", "bold vector art, clean lines, high contrast")
        do_review     = data.get("review", True)
        inspiration   = data.get("inspiration_b64", None)  # base64 image from user

        ant = anthropic.Anthropic(api_key=ant_key)
        oai = openai.OpenAI(api_key=oai_key) if oai_key else None

        emit(job_id, "log", f"🚀 Starting for niche: «{niche}»")
        emit(job_id, "log", f"📐 {count} designs | Mode: {design_mode.upper()} | Review: {'ON' if do_review else 'OFF'}")
        if inspiration:
            emit(job_id, "log", "🖼  Inspiration image loaded — Claude will draw style from it")
        if dry_run:
            emit(job_id, "warn", "⚠️  Dry run — designs won't be uploaded")

        emit(job_id, "step", "Brainstorming concepts…")
        concepts = generate_concepts(ant, niche, count, inspiration)
        emit(job_id, "log", f"✅ Got {len(concepts)} concepts")

        for i, concept in enumerate(concepts, 1):
            if is_cancelled(job_id):
                emit(job_id, "warn", f"🛑 Stopped after {i-1} designs.")
                break

            emit(job_id, "step", f"[{i}/{len(concepts)}] «{concept['slogan']}»")

            # ── Generate design ───────────────────────────────────────────
            file_type = "svg"
            if design_mode == "svg":
                emit(job_id, "log", "  🎨 Generating SVG with Claude…")
                try:
                    svg_code = generate_svg(ant, concept, img_style, inspiration)
                    # Convert SVG to PNG for preview + upload
                    design_bytes, file_type = svg_to_png(svg_code)
                    if file_type == "svg":
                        design_bytes = svg_code.encode("utf-8")
                    emit(job_id, "log", f"  ✅ SVG generated ({file_type.upper()})")
                except Exception as e:
                    emit(job_id, "error", f"  ❌ SVG generation failed: {e}")
                    continue
            else:
                if not oai:
                    emit(job_id, "error", "  ❌ OpenAI key required for DALL·E mode")
                    continue
                emit(job_id, "log", "  🎨 Generating image via DALL·E 3…")
                try:
                    design_bytes = generate_dalle_image(oai, concept, img_style)
                    file_type = "png"
                    emit(job_id, "log", "  ✂️  Removing background…")
                    design_bytes = remove_background_auto(design_bytes)
                except Exception as e:
                    emit(job_id, "error", f"  ❌ Image failed: {e}")
                    continue

            # ── Metadata ──────────────────────────────────────────────────
            emit(job_id, "log", "  📝 Writing metadata…")
            metadata = generate_metadata(ant, niche, concept)
            emit(job_id, "log", f"  📌 {metadata['title']}")

            img_b64 = base64.b64encode(design_bytes).decode()

            # ── REVIEW GATE ──────────────────────────────────────────────
            if do_review:
                review_event = threading.Event()
                jobs[job_id]["review_pending"] = {
                    "image_b64": img_b64,
                    "metadata":  metadata,
                    "slogan":    concept["slogan"],
                    "index":     i,
                    "file_type": file_type,
                    "event":     review_event,
                    "decision":  None
                }
                # Emit lightweight trigger — NO image data in SSE
                emit(job_id, "review", f"👁  Design {i} ready — waiting for your review…")
                review_event.wait(timeout=600)
                decision = jobs[job_id]["review_pending"]["decision"]
                jobs[job_id]["review_pending"] = None

                if not decision or decision.get("action") == "skip":
                    emit(job_id, "warn", f"  ⏭  Skipped design {i}")
                    continue
                if decision.get("metadata"):
                    metadata = decision["metadata"]
                emit(job_id, "log", f"  ✅ Approved: {metadata['title']}")

                # Re-generate if user requested changes
                if decision.get("regenerate"):
                    emit(job_id, "log", f"  🔄 Regenerating based on feedback: {decision.get('feedback','')}")
                    concept["visual_style"] += f". User feedback: {decision.get('feedback','')}"
                    try:
                        svg_code = generate_svg(ant, concept, img_style, inspiration)
                        design_bytes, file_type = svg_to_png(svg_code)
                        if file_type == "svg":
                            design_bytes = svg_code.encode("utf-8")
                        img_b64 = base64.b64encode(design_bytes).decode()
                    except Exception as e:
                        emit(job_id, "warn", f"  ⚠️  Regen failed, using original: {e}")
            # ─────────────────────────────────────────────────────────────

            result = {
                "slogan": concept["slogan"], "title": metadata["title"],
                "description": metadata["description"], "tags": metadata["tags"],
                "image_b64": img_b64, "file_type": file_type,
                "uploaded": False, "upload_error": None
            }

            if not dry_run:
                emit(job_id, "log", "  📤 Uploading to Spreadshirt…")
                try:
                    url = upload_to_spreadshirt(design_bytes, file_type, metadata, ss_key, ss_user, job_id)
                    result["uploaded"] = True
                    result["product_url"] = url
                    emit(job_id, "log", "  ✅ Uploaded!")
                    record_upload(niche, concept["slogan"], metadata["title"], url)
                except Exception as e:
                    result["upload_error"] = str(e)
                    emit(job_id, "error", f"  ❌ Upload failed: {e}")
            else:
                emit(job_id, "log", "  💾 Dry run — skipping upload")

            # Save design to disk
            safe_slogan = re.sub(r'[^a-zA-Z0-9]+', '_', concept["slogan"])[:40]
            filename = f"{job_id}_{i:02d}_{safe_slogan}.{file_type}"
            filepath = os.path.join(OUTPUTS_DIR, filename)
            with open(filepath, "wb") as f:
                f.write(design_bytes)
            result["saved_path"] = filepath
            emit(job_id, "log", f"  💾 Saved: outputs/{filename}")

            jobs[job_id]["results"].append(result)
            emit(job_id, "result", f"Design {i} done", index=len(jobs[job_id]["results"])-1)

        n = len(jobs[job_id]["results"])
        if not is_cancelled(job_id):
            emit(job_id, "log", f"🎉 All done! {n} designs created.")
            jobs[job_id]["status"] = "done"
        else:
            jobs[job_id]["status"] = "cancelled"

    except Exception as e:
        emit(job_id, "error", f"💥 Fatal error: {e}")
        jobs[job_id]["status"] = "error"


# ─── AI Design Helpers ────────────────────────────────────────────────────────

def generate_concepts(ant, niche, count, inspiration_b64=None):
    inspiration_note = ""
    if inspiration_b64:
        inspiration_note = "Also consider the style of the provided inspiration image."

    prompt = f"""You are a creative director for a print-on-demand t-shirt brand.
Generate {count} UNIQUE, marketable t-shirt design concepts for niche: "{niche}".
{inspiration_note}
- Mix funny, inspirational, bold slogans
- SHORT slogans (max 6 words) — punchy and memorable
- Each concept distinct in theme

Return ONLY valid JSON array, no markdown:
[{{"slogan":"...","visual_style":"SVG art direction: shapes, layout, colors","mood":"funny|inspirational|bold|nostalgic|edgy"}}]"""

    messages = [{"role": "user", "content": prompt}]
    if inspiration_b64:
        messages = [{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": inspiration_b64}},
            {"type": "text", "text": prompt}
        ]}]

    resp = ant.messages.create(model="claude-opus-4-6", max_tokens=2000, messages=messages)
    text = resp.content[0].text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"): text = text[4:]
    return json.loads(text.strip())


def generate_svg(ant, concept, style, inspiration_b64=None):
    """Claude generates a complete SVG design — true vector, zero spelling errors."""

    inspiration_note = ""
    if inspiration_b64:
        inspiration_note = "Match the aesthetic style of the provided inspiration image."

    prompt = f"""You are an expert SVG designer for screen-printed t-shirts.
Create a complete, valid SVG for this design:
Slogan: "{concept['slogan']}"
Style: {concept['visual_style']}
Art direction: {style}
{inspiration_note}

STRICT REQUIREMENTS — follow every rule exactly:

CANVAS & LAYOUT:
- viewBox="0 0 1000 1000" width="1000" height="1000"
- NO background rectangle — transparent background only
- Keep ALL elements strictly within x=80 to x=920, y=80 to y=920 (80px safe margin)
- Center the entire composition horizontally and vertically

COLORS — MAXIMUM 3 COLORS TOTAL:
- Choose colors that create strong contrast on BOTH white and black shirts
- Every filled shape MUST have a contrasting stroke (min stroke-width="6") so it reads on any shirt color
- Good combos: black + white + one accent / navy + gold + white / red + black + white
- NO gradients, NO opacity tricks, NO semi-transparent layers

SHAPES & ILLUSTRATION:
- Use BOLD, SIMPLE geometric shapes only — thick strokes, chunky forms
- Aim for 5-15 distinct elements total — less is more
- NO tiny details, NO fine textures, NO hatching
- Illustration must be instantly readable at thumbnail size
- NEVER use <textPath> or arc/curved text of any kind

TYPOGRAPHY:
- font-family="Arial Black, Impact, sans-serif" font-weight="900"
- Text MUST be spelled EXACTLY: "{concept['slogan']}"
- Use straight horizontal <text> elements only — no arcs, no transforms on text
- Minimum font-size 70 for main slogan, bold stroke outline for legibility
- Text stroke: stroke="#000000" or contrasting dark color, stroke-width="8", paint-order="stroke fill"

Return ONLY the raw SVG code starting with <svg, nothing else."""

    messages = [{"role": "user", "content": prompt}]
    if inspiration_b64:
        messages = [{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": inspiration_b64}},
            {"type": "text", "text": prompt}
        ]}]

    resp = ant.messages.create(model="claude-opus-4-6", max_tokens=4000, messages=messages)
    text = resp.content[0].text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("svg"): text = text[3:]
        elif text.startswith("xml"): text = text[3:]
    # Ensure it starts with <svg
    idx = text.find("<svg")
    if idx > 0:
        text = text[idx:]
    return text.strip()


def minify_svg(svg_code):
    """Strip whitespace and comments to reduce SVG file size."""
    import re
    svg_code = re.sub(r'<!--.*?-->', '', svg_code, flags=re.DOTALL)
    svg_code = re.sub(r'\n\s*', ' ', svg_code)
    svg_code = re.sub(r'\s{2,}', ' ', svg_code)
    return svg_code.strip()


def svg_to_png(svg_code, size=1000):
    """Convert SVG bytes to PNG. Returns (bytes, 'png') or (svg_str, 'svg') fallback."""
    if CAIROSVG_AVAILABLE:
        try:
            png_bytes = cairosvg.svg2png(bytestring=svg_code.encode("utf-8"),
                                          output_width=size, output_height=size)
            return png_bytes, "png"
        except Exception:
            pass
    # Fallback: return minified SVG (Spreadshirt accepts SVG)
    return minify_svg(svg_code).encode("utf-8"), "svg"


def generate_dalle_image(oai, concept, style):
    prompt = (
        f'FLAT GRAPHIC ARTWORK ONLY. Text: "{concept["slogan"]}". '
        f'Style: {concept["visual_style"]}. {style}. '
        f'Pure white background. Hard edges. Bold outlines. High contrast. '
        f'NO t-shirt, NO mockup, NO hanger, NO fabric. Just the graphic on white.'
    )
    resp = oai.images.generate(model="dall-e-3", prompt=prompt,
                                size="1024x1024", quality="hd", n=1)
    return requests.get(resp.data[0].url, timeout=30).content


def remove_background_auto(image_bytes):
    if REMBG_AVAILABLE:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
        out = rembg_remove(img)
        r, g, b, a = out.split()
        a = a.point(lambda x: 255 if x > 30 else 0)
        out.putalpha(a)
        buf = io.BytesIO(); out.save(buf, format="PNG"); return buf.getvalue()
    # Simple white removal fallback
    img = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
    data = [(r,g,b,0) if r>230 and g>230 and b>230 else (r,g,b,255) for r,g,b,a in img.getdata()]
    img.putdata(data)
    buf = io.BytesIO(); img.save(buf, format="PNG"); return buf.getvalue()


def generate_metadata(ant, niche, concept):
    prompt = f"""Spreadshirt listing metadata.
Niche: "{niche}", Slogan: "{concept['slogan']}", Mood: {concept.get('mood','bold')}
Return ONLY valid JSON:
{{"title":"SEO title under 60 chars","description":"2-3 sentences keyword-rich","tags":["tag1","tag2","tag3","tag4","tag5","tag6","tag7","tag8","tag9","tag10"]}}"""
    resp = ant.messages.create(model="claude-opus-4-6", max_tokens=600,
                                messages=[{"role":"user","content":prompt}])
    text = resp.content[0].text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"): text = text[4:]
    return json.loads(text.strip())


# ─── Niche Discovery ─────────────────────────────────────────────────────────

def discover_niches(ant, already_used=None, count=10):
    """Ask Claude to suggest fresh, profitable print-on-demand niches."""
    used_note = ""
    if already_used:
        used_note = f"\nALREADY USED — do NOT suggest these: {', '.join(already_used[:30])}"

    prompt = f"""You are a print-on-demand market researcher. Suggest {count} HIGH-POTENTIAL t-shirt niches.

Focus on niches that:
- Have passionate, spending audiences (hobbies, professions, fandoms, lifestyles)
- Are specific enough to have strong identity ("coffee snobs" > "coffee lovers")
- Currently trending or evergreen (pets, fitness, gaming, trades, humor, parenting)
- Work well for slogans + simple graphic designs
{used_note}

Return ONLY a JSON array of strings, no markdown:
["niche 1", "niche 2", ...]"""

    resp = ant.messages.create(
        model="claude-opus-4-6", max_tokens=800,
        messages=[{"role": "user", "content": prompt}]
    )
    text = resp.content[0].text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"): text = text[4:]
    return json.loads(text.strip())


# ─── Autopilot Engine ─────────────────────────────────────────────────────────

def _schedule_next_run():
    """Set next_run timestamp based on interval. Call with _autopilot_lock held."""
    hours = autopilot_state["interval_hours"]
    next_dt = datetime.datetime.utcnow() + datetime.timedelta(hours=hours)
    autopilot_state["next_run"] = next_dt.isoformat()


def _autopilot_runner():
    """Background thread: wakes up, checks if it's time to run, executes a job."""
    while True:
        time.sleep(60)  # check every minute
        with _autopilot_lock:
            if not autopilot_state["enabled"]:
                continue
            next_run = autopilot_state.get("next_run")
            if not next_run:
                _schedule_next_run()
                continue
            if datetime.datetime.utcnow().isoformat() < next_run:
                continue
            # It's time — kick off a run
            autopilot_state["last_run"] = datetime.datetime.utcnow().isoformat()
            _schedule_next_run()
            ant_key = autopilot_state.get("anthropic_key") or os.environ.get("ANTHROPIC_KEY", "")
            if not ant_key:
                continue
            count = autopilot_state["designs_per_run"]
            creds = {k: autopilot_state.get(k, "") for k in
                     ("anthropic_key", "openai_key", "spreadshirt_key", "spreadshirt_user",
                      "design_mode", "style")}

        # Discover a fresh niche (outside lock to avoid holding it during API call)
        try:
            ant = anthropic.Anthropic(api_key=ant_key)
            history = load_history()
            suggestions = discover_niches(ant, history.get("niches_used", []), count=5)
            niche = suggestions[0] if suggestions else "dog lovers"
        except Exception:
            niche = "cat lovers"

        job_id = str(uuid.uuid4())[:8]
        jobs[job_id] = {
            "status": "running", "events": [], "results": [],
            "cancelled": False, "review_pending": None
        }
        job_data = {
            "niche": niche,
            "count": count,
            "anthropic_key": creds.get("anthropic_key") or os.environ.get("ANTHROPIC_KEY", ""),
            "openai_key": creds.get("openai_key") or os.environ.get("OPENAI_KEY", ""),
            "spreadshirt_key": creds.get("spreadshirt_key") or os.environ.get("SPREADSHIRT_KEY", ""),
            "spreadshirt_user": creds.get("spreadshirt_user") or os.environ.get("SPREADSHIRT_USER", ""),
            "design_mode": creds.get("design_mode") or "svg",
            "style": creds.get("style") or "bold vector art, clean lines, high contrast",
            "review": False,  # Always unattended
        }
        threading.Thread(target=_autopilot_job, args=(job_id, job_data, niche), daemon=True).start()

        with _autopilot_lock:
            autopilot_state["runs"].append({
                "timestamp": autopilot_state["last_run"],
                "niche": niche,
                "job_id": job_id,
            })


def _autopilot_job(job_id, data, niche):
    """Run agent then record results into history."""
    run_agent(job_id, data)
    job = jobs.get(job_id, {})
    for result in job.get("results", []):
        record_upload(
            niche=niche,
            slogan=result.get("slogan", ""),
            title=result.get("title", ""),
            product_url=result.get("product_url"),
        )
    # Update run entry with upload count
    with _autopilot_lock:
        for run in autopilot_state["runs"]:
            if run["job_id"] == job_id:
                run["uploads"] = len(job.get("results", []))
                run["status"] = job.get("status", "done")
                break


# Start the autopilot background thread immediately
_ap_thread = threading.Thread(target=_autopilot_runner, daemon=True)
_ap_thread.start()


# ─── Spreadshirt Upload ───────────────────────────────────────────────────────

def upload_to_spreadshirt(design_bytes, file_type, metadata, ss_key, ss_user, job_id):
    """
    Spreadshirt Partner API upload flow.
    Tries multipart POST first, then falls back to JSON with base64.
    """
    auth = (ss_key, "")
    base = SPREADSHIRT_BASE
    mime = "image/svg+xml" if file_type == "svg" else "image/png"
    filename = f"design.{file_type}"

    # ── 1. Upload design ──────────────────────────────────────────────────
    emit(job_id, "log", f"    → Uploading {file_type.upper()} design…")

    # Try multipart first
    resp = requests.post(
        f"{base}/users/{ss_user}/designs",
        auth=auth,
        files={"file": (filename, io.BytesIO(design_bytes), mime)},
        data={"name": metadata["title"][:50]},
        timeout=60
    )
    emit(job_id, "log", f"    → Design HTTP {resp.status_code}")

    # If multipart fails, try JSON+base64
    if resp.status_code in (405, 415, 400):
        emit(job_id, "log", "    → Trying JSON/base64 upload…")
        resp = requests.post(
            f"{base}/users/{ss_user}/designs",
            auth=auth,
            json={
                "name": metadata["title"][:50],
                "resources": [{"type": file_type, "data": base64.b64encode(design_bytes).decode()}]
            },
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            timeout=60
        )
        emit(job_id, "log", f"    → JSON upload HTTP {resp.status_code}")

    if not resp.ok:
        raise Exception(f"Design upload failed ({resp.status_code}): {resp.text[:500]}")

    # Extract design ID
    design_id = None
    ct = resp.headers.get("Content-Type", "")
    if "json" in ct:
        try: design_id = resp.json().get("id")
        except: pass
    if not design_id and resp.text.strip().startswith("<"):
        m = re.search(r'<id[^>]*>(\d+)<\/id>', resp.text)
        if m: design_id = m.group(1)
    if not design_id:
        loc = resp.headers.get("Location", "")
        if loc: design_id = loc.rstrip("/").split("/")[-1]
    if not design_id:
        raise Exception(f"Could not get design ID. Response: {resp.text[:500]}")

    emit(job_id, "log", f"    → Design ID: {design_id}")

    # ── 2. Create product ─────────────────────────────────────────────────
    emit(job_id, "log", "    → Creating product…")
    prod_resp = requests.post(
        f"{base}/users/{ss_user}/products",
        auth=auth,
        json={
            "name": metadata["title"][:60],
            "description": metadata["description"],
            "tags": metadata["tags"],
            "productTypeId": "210",
            "configurations": [{"type":"design","designId":str(design_id),"printAreaId":"1"}]
        },
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        timeout=30
    )
    emit(job_id, "log", f"    → Product HTTP {prod_resp.status_code}")
    if not prod_resp.ok:
        raise Exception(f"Product failed ({prod_resp.status_code}): {prod_resp.text[:500]}")
    return prod_resp.headers.get("Location", "")


if __name__ == "__main__":
    print("=" * 50)
    print("  Spreadshirt AI Design Agent")
    print("  Open http://localhost:5000")
    print("=" * 50)
    if not CAIROSVG_AVAILABLE:
        print("  cairosvg not installed -- SVGs won't be converted to PNG")
        print("     Run: pip install cairosvg")
    app.run(debug=True, port=5000, threaded=True)
