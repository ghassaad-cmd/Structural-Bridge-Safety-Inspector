# 3–5 Minute Demonstration Script

## Before the presentation

- Open the repository in VS Code or GitHub Codespaces.
- Confirm `models/cics_hog_mlp.joblib` is available locally and the dashboard
  dependencies are installed in the selected Python environment.
- Have one JPG/JPEG/PNG concrete image ready. For a known example, use
  `data/CICS/Simple Cracks/1.jpg`; the model prediction is not guaranteed to
  match the folder label.
- In Codespaces, be ready to open the forwarded port **8501**.

## Walkthrough (about 4 minutes)

### 0:00–0:40 — Launch and orient

From the repository root, activate the project virtual environment and run:

```bash
streamlit run app.py --server.address 0.0.0.0
```

Open the local Streamlit page or the Codespaces forwarded port 8501. Point out
the project overview and the persistent prototype / not-certified warning.

### 0:40–1:20 — Upload a crack image

In **Image inspection**, upload the prepared JPG/JPEG/PNG. Show that the image
is previewed beside the model result. Explain that the model uses the saved
artifact and is not retrained on upload.

### 1:20–2:10 — Review classification and preliminary assessment

Point out the predicted class and confidence. Review the preliminary severity,
maintenance priority, and explanation. If the example's confidence is below
60%, show the `Uncertain` / `Manual inspection` prompt. Explain that these are
prototype rules, not engineering criteria or a safety determination.

### 2:10–3:10 — Ask the RAG assistant

Ask a question supported by the local knowledge base, such as:

> How should visible concrete cracks be documented?

Show the retrieved FHWA source reference. Optionally enter an unrelated
question and demonstrate that the app states when information is unavailable
instead of inventing an answer.

### 3:10–4:00 — Download and close

Click **Download inspection report**. Note that the Markdown report contains
the image filename, classification, confidence, preliminary assessment,
timestamp, and limitations. Close by reiterating that the classifier covers
concrete crack categories only, does not detect corrosion, and cannot replace
qualified inspection.

## Demonstration notes

- Inference output can vary by image; do not promise a particular prediction.
- Avoid presenting the demonstration severity or priority as a certified
  engineering result.
- RAG coverage is deliberately limited to the application's curated FHWA
  knowledge entries.
