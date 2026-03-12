"""Contract checks for implementation-plan and ground-truth documentation."""

from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
IMPLEMENTATION_TASKS = REPO_ROOT / "IMPLEMENTATION_TASKS.md"
EXEC_ORDER = REPO_ROOT / "docs" / "CURSOR_EXECUTION_ORDER.md"
GROUND_TRUTH = REPO_ROOT / "docs" / "PROJECT_GROUND_TRUTH.md"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _phase_index(text: str, phase_label: str) -> int:
    idx = text.find(phase_label)
    if idx < 0:
        raise AssertionError(f"Missing phase label: {phase_label}")
    return idx


def test_phase_gates_are_in_coherent_order() -> None:
    text = _read(IMPLEMENTATION_TASKS)
    p3 = _phase_index(text, "# Phase 3. Event normalization and codebooks")
    p35 = _phase_index(text, "# Phase 3.5. Action extraction feasibility checkpoint")
    p4 = _phase_index(text, "# Phase 4. Canonical timeline construction")
    p6 = _phase_index(text, "# Phase 6. Train/validation/test splitting")
    p65 = _phase_index(text, "# Phase 6.5. Output schemas and runner validation gate")
    p7 = _phase_index(text, "# Phase 7. Replay environment and structured tools")
    assert p3 < p35 < p4
    assert p6 < p65 < p7


def test_execution_order_includes_required_gates() -> None:
    text = _read(EXEC_ORDER)
    idx_norm = _phase_index(text, "3. **Normalization codebooks**")
    idx_feas = _phase_index(text, "4. **Action extraction feasibility checkpoint**")
    idx_timeline = _phase_index(text, "5. **Canonical timeline generator**")
    idx_schema = _phase_index(text, "8. **Output schema freezing and runner validation**")
    idx_replay = _phase_index(text, "9. **Replay environment and structured tools**")
    idx_baselines = _phase_index(text, "10. **Baselines**")
    idx_agent = _phase_index(text, "11. **Gemini constrained agent**")
    assert idx_norm < idx_feas < idx_timeline
    assert idx_schema < idx_replay < idx_baselines < idx_agent


def test_no_langchain_v1_constraint_present() -> None:
    impl = _read(IMPLEMENTATION_TASKS).lower()
    gt = _read(GROUND_TRUTH).lower()
    assert "no langchain in v1" in impl
    assert "langchain" in gt and "out of scope for v1" in gt


def test_feature_builder_contract_present() -> None:
    impl = _read(IMPLEMENTATION_TASKS).lower()
    assert "feature builder contract for tabular ml" in impl
    assert "train only" in impl
    assert "feature_spec_version" in impl


def test_negative_window_non_leak_language_consistent() -> None:
    impl = _read(IMPLEMENTATION_TASKS).lower()
    exec_order = _read(EXEC_ORDER).lower()
    assert "label-generation-only" in impl
    assert "must never be exposed" in impl
    assert "negative-window future logic strictly internal" in exec_order
    assert "never surface negative-window-only future logic" in exec_order


def test_no_external_repo_path_dependency_in_plan() -> None:
    impl = _read(IMPLEMENTATION_TASKS).lower()
    assert "external_repos/" not in impl
