import pytest

from pour_decisions.matrix.registry import MATRIX, by_key, select


def test_matrix_package_imports():
    import pour_decisions.matrix  # noqa: F401


def test_matrix_keys_and_ids_are_unique():
    keys = [s.key for s in MATRIX]
    ids = [s.hf_id for s in MATRIX]
    assert len(keys) == len(set(keys))
    assert len(ids) == len(set(ids))


def test_gated_implies_open_weight_tier():
    for s in MATRIX:
        if s.gated:
            assert s.license_tier == "open-weight"


def test_base_models_have_no_chat_template():
    # v1 matrix is base-only
    assert all(s.has_chat_template is False for s in MATRIX)


def test_mistral_entry_present():
    s = by_key("ministral-3-3b-base")
    assert s.hf_id == "mistralai/Ministral-3-3B-Base-2512"
    assert s.license_tier == "free"
    assert s.gated is False


def test_by_key_unknown_raises():
    with pytest.raises(KeyError):
        by_key("does-not-exist")


def test_select_excludes_gated_when_requested():
    free_only = select(tiers={"free"}, include_gated=False)
    assert all(s.license_tier == "free" and not s.gated for s in free_only)
    assert by_key("gemma-3-270m") not in free_only
