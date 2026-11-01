from __future__ import annotations

from datetime import datetime
from io import BytesIO
from pathlib import Path
import tempfile

import streamlit as st
from PIL import Image, UnidentifiedImageError

from bridge_safety_inspector.health import (
    CrackClassificationAssessment,
    assess_crack_classification,
)
from bridge_safety_inspector.model import TrainedCrackClassifier
from bridge_safety_inspector.rag import DEFAULT_BRIDGE_KB, RAGAnswer
from bridge_safety_inspector.report import build_inspection_report


PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "models" / "cics_hog_mlp.joblib"
ALLOWED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


@st.cache_resource(show_spinner="Loading the trained crack classifier...")
def load_classifier() -> TrainedCrackClassifier:
    import joblib

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Trained model not found at {MODEL_PATH}. "
            "Place the existing cics_hog_mlp.joblib model in models/."
        )
    return joblib.load(MODEL_PATH)


def predict_uploaded_image(
    classifier: TrainedCrackClassifier, image_bytes: bytes, suffix: str
) -> tuple[str, float]:
    with tempfile.NamedTemporaryFile(suffix=suffix) as image_file:
        image_file.write(image_bytes)
        image_file.flush()
        return classifier.predict_with_confidence([image_file.name])[0]


def render_result(
    image_filename: str,
    category: str,
    confidence: float,
    assessment: CrackClassificationAssessment,
) -> None:
    st.subheader("Classification")
    st.metric("Predicted category", category)
    st.progress(confidence, text=f"Prediction confidence: {confidence:.1%}")

    st.subheader("Preliminary health assessment")
    first_column, second_column = st.columns(2)
    first_column.metric("Preliminary severity", assessment.severity_level)
    second_column.metric("Maintenance priority", assessment.maintenance_priority)
    st.write(assessment.explanation)
    if assessment.manual_inspection_required:
        st.warning(
            "Manual inspection is required. Do not rely on this low-confidence "
            "image classification."
        )
    else:
        st.info(
            "This preliminary category-based indication is not a certified "
            "engineering assessment."
        )

    report_time = datetime.now().astimezone()
    report = build_inspection_report(
        image_filename=image_filename,
        predicted_category=category,
        confidence=confidence,
        assessment=assessment,
        inspected_at=report_time,
    )
    st.download_button(
        "Download inspection report",
        data=report,
        file_name=f"inspection-report-{report_time:%Y%m%d-%H%M%S}.md",
        mime="text/markdown",
        type="primary",
    )


def render_knowledge_assistant() -> None:
    st.header("RAG knowledge assistant")
    st.caption(
        "Ask about concrete cracks, bridge inspection, inspection procedures, "
        "maintenance, or preservation. Answers use the local FHWA reference "
        "knowledge base; no external AI service is called."
    )
    with st.form("knowledge_question_form"):
        question = st.text_input(
            "Your question",
            placeholder="How should visible concrete cracks be documented?",
        )
        submitted = st.form_submit_button("Ask the knowledge base", type="primary")

    if submitted:
        if not question.strip():
            st.warning("Enter a question to search the available references.")
        else:
            st.session_state["rag_answer"] = DEFAULT_BRIDGE_KB.answer(
                question.strip()
            )

    answer = st.session_state.get("rag_answer")
    if isinstance(answer, RAGAnswer):
        if answer.sources:
            st.markdown(answer.answer)
            st.markdown("**Supporting FHWA references**")
            for source in answer.sources:
                st.markdown(f"- [{source.title}]({source.url})")
        else:
            st.info(
                "The requested information is unavailable in the current "
                f"knowledge base. {answer.answer}"
            )


def main() -> None:
    st.set_page_config(
        page_title="Automated Structural Bridge Safety Inspector",
        page_icon="🌉",
        layout="wide",
    )
    st.markdown(
        """
        <style>
        .block-container { max-width: 1180px; padding-top: 2rem; }
        [data-testid="stMetric"] {
            background: #f3f6f8; border: 1px solid #dfe6eb;
            padding: 1rem; border-radius: 0.65rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.title("Automated Structural Bridge Safety Inspector")
    st.caption(
        "A presentation-ready prototype for concrete crack image "
        "classification and sourced inspection references."
    )
    st.warning(
        "Prototype only — this application is not a certified engineering "
        "safety assessment and must not replace qualified bridge inspection."
    )

    st.header("Project overview")
    overview = st.columns(3)
    overview[0].metric("Image classes", "3 concrete-crack categories")
    overview[1].metric("Assessment", "Preliminary only")
    overview[2].metric("Knowledge sources", "FHWA references")
    st.write(
        "Upload one JPG, JPEG, or PNG image for model classification. The "
        "dashboard reports model confidence and a preliminary demonstration "
        "assessment, and provides a separate question-answering assistant "
        "grounded in the project's local knowledge base."
    )

    st.header("Image inspection")
    uploaded_image = st.file_uploader(
        "Upload a concrete inspection image",
        type=["jpg", "jpeg", "png"],
        help="Supported formats: JPG, JPEG, and PNG.",
    )

    if uploaded_image is not None:
        image_suffix = Path(uploaded_image.name).suffix.lower()
        if image_suffix not in ALLOWED_IMAGE_SUFFIXES:
            st.error("Choose an image with a JPG, JPEG, or PNG file extension.")
        else:
            image_bytes = uploaded_image.getvalue()
            try:
                with Image.open(BytesIO(image_bytes)) as image:
                    image.verify()
            except (
                Image.DecompressionBombError,
                Image.DecompressionBombWarning,
                OSError,
                UnidentifiedImageError,
                ValueError,
            ) as error:
                st.error(
                    "The uploaded file is not a valid, readable image. "
                    f"Choose another JPG, JPEG, or PNG file. Details: {error}"
                )
            else:
                image_column, results_column = st.columns([1.1, 1])
                with image_column:
                    st.image(
                        image_bytes,
                        caption=uploaded_image.name,
                    )

                with results_column:
                    try:
                        classifier = load_classifier()
                        category, confidence = predict_uploaded_image(
                            classifier, image_bytes, image_suffix
                        )
                        assessment = assess_crack_classification(
                            category, confidence
                        )
                    except Exception as error:
                        st.error(
                            "The image could not be classified. Confirm the "
                            "trained model and its dependencies are available, "
                            f"then try again. Details: {error}"
                        )
                    else:
                        render_result(
                            uploaded_image.name,
                            category,
                            confidence,
                            assessment,
                        )

    render_knowledge_assistant()
    st.divider()
    st.caption(
        "Scope limitation: the trained model classifies concrete crack images "
        "only; corrosion detection is not supported."
    )


if __name__ == "__main__":
    main()
