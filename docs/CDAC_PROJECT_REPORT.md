# Automated Structural Bridge Safety Inspector

## CDAC Project Report

**Project type:** Demonstration prototype
**Primary workflow:** Concrete crack image classification, preliminary assessment, and sourced reference retrieval
**Safety status:** Not a certified engineering safety assessment

> This report describes implemented software and the recorded model
> evaluation. It does not make bridge-safety claims or provide engineering
> advice.

## 1. Introduction

This project combines an image-classification model, a lightweight
rule-based health assessment, a source-grounded reference assistant, and a
Streamlit user interface. An inspector can upload a concrete image, view the
model's predicted crack category and confidence, review preliminary
demonstration guidance, ask a question against the local FHWA knowledge base,
and download a Markdown report.

The project is a prototype for demonstrating software integration. It is not
a replacement for a bridge inspection program or qualified engineering review.

## 2. Problem statement

Inspection workflows involve reviewing visual observations and consulting
reference material. This project explores a compact, demonstrable software
workflow to classify a limited set of concrete crack images and expose related
reference information. The scope is deliberately limited: image predictions
do not establish structural condition or safety, and the knowledge assistant
does not contain project-specific inspection records.

## 3. Objectives

Implemented objectives:

1. Load and use the existing trained image classifier without retraining.
2. Predict one of three concrete-image categories and show its model
   probability.
3. Map model output to a clearly labeled preliminary health prompt, with a
   manual-inspection gate for low-confidence predictions.
4. Retrieve relevant, sourced FHWA reference entries and report when no
   information is available.
5. Present the workflow in a laptop-friendly Streamlit dashboard and generate
   a downloadable inspection summary.
6. Preserve transparent limitations and avoid claims of corrosion detection
   or certified bridge assessment.

## 4. Dataset and preprocessing

The dataset is organized as JPEG image directories named `Multibranched
Crack`, `Simple Cracks`, and `Without Crack`. The local image collection has
4,000 files per class (12,000 files total). The verified source is *Cracks In
Concrete Structures (CICS) Dataset* by Hamed Jabbari, Nooshin Bigdeli, and
Mahdi Shojaei, published by Mendeley Data on 16 June 2023 (version 1,
[doi:10.17632/9brnm3c39k.1](https://doi.org/10.17632/9brnm3c39k.1)). Its
record describes 12,000 grayscale 330 x 330 patches derived from concrete
structure imagery, visually assigned to the three categories. DataCite
metadata lists CC BY 4.0; attribution and license terms should be preserved
when redistributing the dataset.

The dataset splitter uses SHA-256 content hashes to de-duplicate images and
avoid separating exact duplicates across splits. Identical images with
conflicting class labels are excluded. With seed 42, the saved evaluation
records 4,064 training, 870 validation, and 874 test samples, and 20
conflicting-label image files excluded. The extracted image dataset and source
archive are local assets and are excluded from Git.

Each input is converted to grayscale and resized to 64 x 64 pixels. A HOG
descriptor is computed with 8 orientations, 8 x 8 pixel cells, 2 x 2 cell
blocks, L2-Hys normalization, and a feature vector. The feature scaler is
fitted on the training data and then applied to validation and test data.

## 5. Model architecture and training

The existing trained model is a scikit-learn `MLPClassifier` with one hidden
layer of 64 ReLU units and an Adam optimizer. Training uses mini-batches of up
to 64, balanced by class through resampling, with a learning rate of 0.001.
Validation macro-F1 drives early stopping (up to 12 epochs, patience 3).

The recorded run completed 6 epochs and restored the checkpoint from epoch 3.
The deployable inference artifact is `models/cics_hog_mlp.joblib`, which
contains the trained classifier, scaler, and image size. The dashboard loads
this artifact; Stage 6 did not retrain or modify it.

## 6. Evaluation results

The following values are transcribed from
`models/cics_hog_mlp.metrics.json`. The held-out test split contains 874
samples. Values are dataset results, not a guarantee of field performance.

| Class | Support | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Multibranched Crack | 82 | 69.86% | 62.20% | 65.81% |
| Simple Cracks | 390 | 84.16% | 83.08% | 83.61% |
| Without Crack | 402 | 87.98% | 91.04% | 89.49% |
| **Overall accuracy** | **874** | — | — | **84.78%** |
| **Macro average** | **874** | **80.67%** | **78.77%** | **79.64%** |

Confusion matrix, with rows as actual classes and columns as predictions in
the order Multibranched Crack, Simple Cracks, Without Crack:

```text
[[51, 29,  2],
 [18, 324, 48],
 [ 4, 32, 366]]
```

The lower recorded recall for Multibranched Crack (62.20%) is important when
interpreting the prototype. The evaluation is limited to the saved split and
does not establish generalization to field imagery.

## 7. Bridge health assessment

The implemented `assess_crack_classification()` function maps image categories
to preliminary labels: Without Crack → Low (preliminary) / Routine review;
Simple Cracks → Moderate (preliminary) / Scheduled review; and Multibranched
Crack → High (preliminary) / Prompt review.

At model confidence below 0.60, the function instead returns `Uncertain`,
prioritizes `Manual inspection`, and explains that the category should not be
relied upon. The threshold and mappings are demonstration logic, not
validated engineering thresholds or a condition-rating methodology. Even a
high-confidence output is not a safety conclusion.

## 8. RAG implementation

`DEFAULT_BRIDGE_KB` contains a small, curated in-memory set of entries about
concrete crack observations, monitoring and inspection records, inspection
procedures/quality, and maintenance/preservation. Retrieval normalizes tokens
and ranks entries by overlap across the query and entry title/tags/content.
Answers are composed from retrieved entries only when they have a named
HTTP(S) source. Source names and URLs are included with the answer.

If no eligible entry matches, the assistant returns an explicit
no-information response. The implementation is deterministic lexical
retrieval with extractive/template text: there is no generative language model,
external AI call, full-text FHWA document index, or project-specific asset
database.

The curated references are:

- FHWA, [Bridge Inspector's Reference Manual](https://www.fhwa.dot.gov/bridge/nbis/pubs/nhi21002.pdf).
- FHWA, [Recommended Framework for a Bridge Inspection QC/QA Program](https://www.fhwa.dot.gov/bridge/nbis/nbisframework.cfm).
- FHWA, [Bridge Preservation Guide](https://www.fhwa.dot.gov/bridge/preservation/guide/guide.pdf).

These references provide general information only; the application does not
claim that the RAG component constitutes a complete or authoritative
interpretation of those documents.

## 9. Streamlit dashboard

`app.py` provides:

- JPG, JPEG, and PNG upload, format validation, and image display.
- Prediction using the saved model and display of predicted category and
  probability.
- Preliminary severity, maintenance priority, explanation, and low-confidence
  manual-inspection warning.
- A question form for the local RAG knowledge base, with source links or an
  explicit unavailable-information message.
- A downloadable Markdown report including filename, prediction, confidence,
  assessment, timestamp, and prototype limitations.

The interface states that it is not a certified engineering safety
assessment. The model covers concrete crack categories only and does not
detect corrosion.

## 10. Testing and results

The project test suite covers deterministic dataset splitting, image feature
extraction and training behavior, health assessment, RAG retrieval and
citations, model inference confidence, and report generation. Final Stage 6
verification completed with **32 tests passing**.

Additional verification confirmed that:

- the existing `.joblib` model loads and exposes all three trained classes;
- a sample image from each category runs through the dashboard inference path
  and returns a supported category and confidence in the 0–1 range;
- the dashboard renders using Streamlit's app test runner;
- a relevant RAG question returns a sourced FHWA answer and an unrelated
  question produces an explicit no-information response;
- the report includes its required result, timestamp, and limitation fields;
- the Streamlit server starts and its health endpoint responds `ok`.

The sample-image check verifies successful inference, not classification
accuracy on arbitrary uploaded images. The metrics table above is the
quantitative evaluation.

## 11. Limitations

- This is a prototype and is not certified, validated, or approved for
  operational bridge safety decisions.
- The classifier is limited to three concrete crack-image categories. It does
  not detect corrosion, identify bridge elements, measure crack width, monitor
  change, or assess structural capacity.
- A model probability is not calibrated certainty or a probability of bridge
  safety. The demonstration assessment threshold is not an engineering
  standard.
- The recorded test metrics come from one dataset split, with lower recall for
  Multibranched Crack; field-domain performance is unknown.
- The RAG knowledge base is small and curated, and lexical token overlap can
  miss relevant material or retrieve incomplete context.
- The report is a generated image-level summary, not a signed inspection
  record, bridge inventory, maintenance plan, or repair design.
- The dataset provenance and redistribution terms should be confirmed against
  its original provider record.

## 12. Future improvements

The following are **not implemented**:

- Obtain independent field images and expert-reviewed labels; evaluate on
  external data and document capture conditions.
- Assess model calibration, robustness, failure modes, and class-wise
  performance before expanding use.
- Introduce qualified human review, audit trails, and role-appropriate access
  if a real operational workflow is ever considered.
- Expand and evaluate reference retrieval, retain source provenance, and
  index full approved documents.
- Integrate with bridge-owner systems only after privacy, security, and
  engineering validation.

## 13. Conclusion

The completed prototype demonstrates an end-to-end software path from a
concrete image through saved-model classification, preliminary category-based
health prompts, and an exportable report, alongside a separately grounded
FHWA reference assistant. The recorded held-out test accuracy is 84.78% and
macro-F1 is 79.64%; class-wise results show meaningful variation. These
results and rules are insufficient for safety determinations. Qualified
inspection and engineering judgment remain essential.

## 14. References

1. Federal Highway Administration, [Bridge Inspector's Reference Manual](https://www.fhwa.dot.gov/bridge/nbis/pubs/nhi21002.pdf).
2. Federal Highway Administration, [Recommended Framework for a Bridge Inspection QC/QA Program](https://www.fhwa.dot.gov/bridge/nbis/nbisframework.cfm).
3. Federal Highway Administration, [Bridge Preservation Guide](https://www.fhwa.dot.gov/bridge/preservation/guide/guide.pdf).
4. H. Jabbari, N. Bigdeli, and M. Shojaei, *Cracks In Concrete Structures
   (CICS) Dataset*, Mendeley Data, version 1, published 16 June 2023,
   [doi:10.17632/9brnm3c39k.1](https://doi.org/10.17632/9brnm3c39k.1).
   Dataset record: https://data.mendeley.com/datasets/9brnm3c39k/1.
   DataCite metadata lists CC BY 4.0:
   https://creativecommons.org/licenses/by/4.0/.
