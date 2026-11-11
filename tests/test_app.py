from pathlib import Path

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]


def test_streamlit_shell_renders_all_four_views_without_exceptions(monkeypatch) -> None:
    monkeypatch.setenv("GENERATOR_COPILOT_ROOT", str(ROOT))
    monkeypatch.setenv("MPLCONFIGDIR", str(ROOT / ".mplconfig"))

    app = AppTest.from_file(ROOT / "src" / "generator_service_copilot" / "app.py").run(timeout=30)

    assert not app.exception
    assert [item.value for item in app.title] == ["Generator Service Copilot"]
    assert [item.label for item in app.tabs] == ["Ask", "Inspect", "Prompt Lab", "Insights"]
