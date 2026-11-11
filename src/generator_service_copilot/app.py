from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pandas as pd
from redis import Redis

from generator_service_copilot.service import CopilotService


def project_root() -> Path:
    return Path(os.getenv("GENERATOR_COPILOT_ROOT", Path.cwd())).resolve()


def build_service(root: Path) -> CopilotService:
    redis_url = os.getenv("REDIS_URL")
    client = Redis.from_url(redis_url, decode_responses=False) if redis_url else None
    return CopilotService(root, redis_client=client)


def _render_ask(service: CopilotService) -> None:
    import streamlit as st

    st.subheader("Ask approved guidance")
    st.caption("Answers are bounded to the fictional, versioned demo corpus.")
    question = st.text_area(
        "Service question",
        placeholder="The cable insulation looks damaged. What documented path applies?",
    )
    if st.button("Analyze question", type="primary", disabled=not question.strip()):
        with st.spinner("Routing, retrieving, and generating…"):
            result = service.ask(question)
        response = result["response"]
        if response["escalate"]:
            st.warning("Escalation required")
        st.markdown(f"**{response['summary']}**")
        for action in response["actions"]:
            st.write(f"- {action}")
        metrics = st.columns(3)
        metrics[0].metric("Intent", result["intent"].replace("_", " ").title())
        metrics[1].metric("Confidence", f"{result['intent_confidence']:.0%}")
        metrics[2].metric("Cache", result["cache_status"])
        with st.expander("Cited evidence", expanded=True):
            for item in result["evidence"]:
                marker = "✓" if item["chunk_id"] in response["citations"] else "·"
                st.markdown(f"{marker} `{item['chunk_id']}` — {item['document_id']}")
                st.caption(item["text"])
        st.caption(
            f"Retrieval: {result['retrieval_mode']} · Provider: {result['provider_status']} · "
            "Unsupported or low-confidence output is escalated."
        )


def _render_inspect(service: CopilotService) -> None:
    import matplotlib.pyplot as plt
    import streamlit as st

    st.subheader("Inspect cable appearance")
    st.caption("Industrial cable benchmark proxy—not validation on generator equipment.")
    uploaded = st.file_uploader("Inspection image", type=["png", "jpg", "jpeg"])
    if uploaded and st.button("Run visual inspection", type="primary"):
        suffix = Path(uploaded.name).suffix.lower()
        with tempfile.NamedTemporaryFile(suffix=suffix) as temporary:
            temporary.write(uploaded.getvalue())
            temporary.flush()
            with st.spinner("Extracting patch features…"):
                result = service.inspect_image(Path(temporary.name))
        left, right = st.columns(2)
        left.image(uploaded, caption="Uploaded image", width="stretch")
        figure, axis = plt.subplots(figsize=(5, 5))
        axis.imshow(result["anomaly_map"], cmap="inferno")
        axis.axis("off")
        right.pyplot(figure, width="stretch")
        state = "Anomaly signal" if result["is_anomaly"] else "Below anomaly threshold"
        st.metric(state, f"{result['score']:.3f}", delta=f"threshold {result['threshold']:.3f}")
        st.warning(result["disclaimer"])


def _render_prompt_lab(root: Path) -> None:
    import streamlit as st

    st.subheader("Prompt lab")
    report = pd.read_json(root / "artifacts" / "prompts" / "report.json", typ="series")
    variants = pd.DataFrame(report["variants"])
    display = variants[
        [
            "variant_id",
            "schema_validity",
            "citation_validity",
            "grounded_action_coverage",
            "action_consistency",
            "score",
            "passes_safety_gate",
        ]
    ]
    st.dataframe(display, hide_index=True, width="stretch")
    st.success(f"Selected: {report['selected_variant']}")
    st.caption("96 development generations; final service test split remains sealed.")


def _render_insights(root: Path) -> None:
    import streamlit as st

    st.subheader("Service-pattern insights")
    composition = pd.read_csv(root / "artifacts" / "nlp" / "cluster_composition.csv")
    representatives = pd.read_csv(root / "artifacts" / "nlp" / "cluster_representatives.csv")
    st.caption("Patterns come from fictional demo cases and do not describe company operations.")
    st.bar_chart(composition.set_index("cluster"))
    st.dataframe(
        representatives[["cluster", "intent", "members", "text"]],
        hide_index=True,
        width="stretch",
    )


def render_app() -> None:
    import streamlit as st

    st.set_page_config(page_title="Generator Service Copilot", page_icon="⚙️", layout="wide")
    st.markdown(
        """
        <style>
        .stApp { background: #f5f7f4; color: #17221d; }
        [data-testid="stHeader"] { background: rgba(245,247,244,.92); }
        .block-container { max-width: 1180px; padding-top: 2.4rem; }
        h1, h2, h3 { color: #123d2e; letter-spacing: -0.02em; }
        [data-testid="stMetric"] { background: white; border: 1px solid #dce5df;
          border-radius: 12px; padding: 14px; }
        .stButton > button[kind="primary"] { background: #167052; border-color: #167052; }
        </style>
        """,
        unsafe_allow_html=True,
    )
    root = project_root()
    os.environ.setdefault("MPLCONFIGDIR", str(root / ".mplconfig"))

    @st.cache_resource
    def cached_service(root_string: str) -> CopilotService:
        return build_service(Path(root_string))

    service = cached_service(str(root))
    st.title("Generator Service Copilot")
    st.caption("Local, grounded decision support for fictional generator-service workflows")
    status = service.health()
    with st.sidebar:
        st.header("System")
        st.write(f"Redis: `{status['redis']}`")
        st.write(f"Generation: `{status['generation']}`")
        st.info("Decision support only. Qualified review remains authoritative.")
    ask, inspect, prompt_lab, insights = st.tabs(["Ask", "Inspect", "Prompt Lab", "Insights"])
    with ask:
        _render_ask(service)
    with inspect:
        _render_inspect(service)
    with prompt_lab:
        _render_prompt_lab(root)
    with insights:
        _render_insights(root)


def main() -> None:
    from streamlit.web import cli

    app_path = Path(__file__).resolve()
    cli.main_run([str(app_path)])


if __name__ == "__main__":
    render_app()
