"""Single-task ablation support in backend/ml/multitask.py (no model download)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from backend.ml import multitask


def tiny_encoder():
    config = transformers.DistilBertConfig(vocab_size=100, dim=16, n_layers=1, n_heads=2, hidden_dim=32)
    return transformers.DistilBertModel(config)


@pytest.mark.parametrize("tasks,heads", [(("intent", "emotion"), {"intent", "emotion"}),
                                         (("intent",), {"intent"}), (("emotion",), {"emotion"})])
def test_model_builds_only_the_requested_heads(tasks, heads):
    model = multitask.MultiTaskModel(tiny_encoder(), 7, 6, tasks=tasks)
    assert set(model.heads) == heads
    counts = multitask.parameter_counts(model)
    assert counts["total"] == counts["encoder"] + sum(counts["heads"].values())


def test_single_task_heads_get_the_same_updates_as_in_the_joint_run():
    sizes = {"intent": 39861, "emotion": 31979}  # the Colab training split sizes
    per_head = multitask.updates_per_head(sizes, 32)
    assert per_head == 1246  # ceil(39861 / 32): the smaller emotion set is cycled
    joint_steps = len(multitask.TASK_CHOICES["both"]) * per_head
    single_steps = len(multitask.TASK_CHOICES["emotion"]) * per_head
    assert joint_steps / 2 == single_steps  # emotion head: 1246 updates per epoch either way


def test_ablation_runs_never_default_to_the_served_model_dir():
    both = multitask.parse_args([])
    assert both.tasks == "both" and Path(both.out) == multitask.DEFAULT_OUT and both.results_name == "multitask"
    for task in ("intent", "emotion"):
        args = multitask.parse_args(["--tasks", task])
        assert Path(args.out).name == f"ablation_{task}" and Path(args.out) != multitask.DEFAULT_OUT
        assert args.results_name == f"ablation_{task}"


def test_save_and_load_round_trip_keeps_single_task_heads(tmp_path):
    tok = transformers.PreTrainedTokenizerFast(
        tokenizer_object=__import__("tokenizers").Tokenizer(__import__("tokenizers").models.WordLevel({"[UNK]": 0}, unk_token="[UNK]")),
        unk_token="[UNK]")
    model = multitask.MultiTaskModel(tiny_encoder(), 7, 6, tasks=("emotion",))
    multitask.save_model(model, tok, {"intent": list("abcdefg"), "emotion": list("uvwxyz")}, tmp_path, "tiny")
    loaded, _, labels = multitask.load_model(tmp_path)
    assert set(loaded.heads) == {"emotion"} and labels["tasks"] == ["emotion"]
