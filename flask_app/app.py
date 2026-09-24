# from wordcloud import WordCloud
import mlflow
from dotenv import load_dotenv
from flask import Flask, jsonify, request
from flask_cors import CORS

load_dotenv()

app = Flask(__name__)
CORS(app)

# The Chrome extension is the real front end. This page exists so a reviewer can
# exercise the model without installing anything — it posts to /predict, same as
# the extension does. Plain triple-quoted string, NOT an f-string: the CSS braces
# below would break f-string parsing.
DEMO_PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>YouTube Comment Sentiment - model backend</title>
<style>
  :root{ --bg:#fff; --fg:#16181d; --muted:#5b6472; --line:#e3e7ec; --card:#f7f9fb; --link:#1a4d8f; }
  @media (prefers-color-scheme:dark){
    :root{ --bg:#0f1115; --fg:#e6e9ee; --muted:#9aa4b2; --line:#252a33; --card:#161a21; --link:#7fb2f0; }
  }
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--fg);
       font:16px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
  .wrap{max-width:680px;margin:0 auto;padding:48px 16px 72px}
  h1{font-size:1.5rem;margin:0 0 .3em;letter-spacing:-.01em}
  .lede{color:var(--muted);margin:0 0 1.8em}
  a{color:var(--link)}
  textarea{width:100%;min-height:130px;padding:12px;border:1px solid var(--line);
           border-radius:8px;background:var(--card);color:var(--fg);
           font:14px/1.5 ui-monospace,Menlo,Consolas,monospace;resize:vertical}
  button{margin-top:12px;padding:10px 20px;border:0;border-radius:8px;background:#0f7b52;
         color:#fff;font-size:15px;font-weight:600;cursor:pointer}
  button:disabled{opacity:.6;cursor:default}
  #status{margin-top:12px;color:var(--muted);font-size:.9rem;min-height:1.4em}
  .row{display:flex;gap:10px;align-items:flex-start;padding:10px 0;border-bottom:1px solid var(--line)}
  .pill{flex:none;padding:2px 10px;border-radius:99px;font-size:.78rem;font-weight:600;color:#fff}
  .pos{background:#0f7b52} .neg{background:#b3261e} .neu{background:#6b7280}
  footer{margin-top:2.5em;padding-top:1.2em;border-top:1px solid var(--line);
         font-size:.88rem;color:var(--muted)}
</style>
</head>
<body><div class="wrap">

  <h1>YouTube Comment Sentiment</h1>
  <p class="lede">This is the <strong>model backend</strong>. The product is a Chrome
     extension that reads a video's comment section and renders the breakdown in place;
     this page is here so you can test the classifier itself without installing anything.
     One comment per line.</p>

  <textarea id="input">this is exactly what I needed, brilliant explanation
honestly the worst audio I have heard on a tutorial
what version are you running here
boring and too long
the code is at 4:32 for anyone looking</textarea>
  <button id="go">Analyse</button>
  <div id="status"></div>
  <div id="out"></div>

  <footer>
    <strong>What this project demonstrates:</strong> a 5-stage reproducible DVC pipeline,
    MLflow model registry with an accuracy gate blocking promotion below 0.80, containerised
    serving, and GitHub Actions deploying to Azure pinned to the commit SHA.<br><br>
    The model is the smallest part, on purpose: TF-IDF + logistic regression, picked for fast
    inference and a small footprint inside a browser plugin. Trained on Reddit comments and
    applied to YouTube, so it reads explicit sentiment well and understated negativity less
    well. Swapping in a transformer is a training-data and compute question, not an
    architecture change - the pipeline around it is unchanged.<br><br>
    <a href="https://github.com/Roy7721/yt_comment_analysis">Pipeline &amp; training repo</a> &middot;
    <a href="https://github.com/Roy7721/Chrome_plugin">Chrome extension</a>
  </footer>

</div>
<script>
const LABELS = {"1":["positive","pos"], "0":["neutral","neu"], "-1":["negative","neg"]};
const btn = document.getElementById("go");
const status = document.getElementById("status");
const out = document.getElementById("out");

btn.onclick = async () => {
  const comments = document.getElementById("input").value
        .split("\\n").map(s => s.trim()).filter(Boolean);
  if (!comments.length) { status.textContent = "Add at least one comment."; return; }

  out.innerHTML = ""; btn.disabled = true; status.textContent = "Analysing...";
  // The container scales to zero, so a first request after idle pays a wake-up.
  const slow = setTimeout(() => {
    status.textContent = "Still waking up - this runs on a scale-to-zero container, ~20s.";
  }, 3000);

  try {
    const r = await fetch("/predict", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({comments: comments})
    });
    if (!r.ok) throw new Error("API returned " + r.status);
    const data = await r.json();
    out.innerHTML = data.map(d => {
      const [text, cls] = LABELS[String(d.sentiment)] || ["unknown","neu"];
      const safe = d.comment.replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");
      return '<div class="row"><span class="pill ' + cls + '">' + text + '</span><span>' + safe + '</span></div>';
    }).join("");
    status.textContent = data.length + " comment(s) classified.";
  } catch (e) {
    status.textContent = "Failed: " + e.message;
  } finally {
    clearTimeout(slow); btn.disabled = false;
  }
};
</script>
</body></html>"""


@app.route("/")
def home():
    return DEMO_PAGE


# DagShub / MLflow registry
mlflow.set_tracking_uri("https://dagshub.com/Roy7721/yt_comment_analysis.mlflow")

MODEL_URI = "models:/yt_chrome_plugin_model@staging"
# Loaded ONCE at import/startup — this call runs the pyfunc's load_context,
# which loads spaCy + NLTK + the vectorizer + classifier and holds them in memory.
# Every request reuses this warm object; nothing reloads per-request.
print(f"Loading model from {MODEL_URI} ...")
model = mlflow.pyfunc.load_model(MODEL_URI)
print("Model loaded and ready.")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    comments = data.get("comments") if data else None

    # Validate: must be a non-empty list of comment strings
    if not comments or not isinstance(comments, list):
        return jsonify({"error": 'Send JSON like {"comments": ["text", ...]}'}), 400

    try:
        preds = model.predict(comments)  # pyfunc takes a raw list -> array of labels
    except Exception as e:
        app.logger.error(f"Prediction failed: {e}")
        return jsonify({"error": str(e)}), 500

    # Pair each comment with its label. int(p) is REQUIRED — see note below.
    response = [{"comment": c, "sentiment": int(p)} for c, p in zip(comments, preds)]
    return jsonify(response)


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False, port=5000)
