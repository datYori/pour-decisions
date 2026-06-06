from pathlib import Path

from pour_decisions.matrix.registry import by_key
from pour_decisions.matrix.trainer import mlx_lora_command


def test_mlx_command_has_required_flags():
    spec = by_key("qwen2.5-0.5b")
    cmd = mlx_lora_command(spec, Path("/d/data"), Path("/d/adapter"))
    assert cmd[:3] == ["mlx_lm.lora", "--model", "Qwen/Qwen2.5-0.5B"]
    assert "--train" in cmd
    assert "--data" in cmd and "/d/data" in cmd
    assert "--adapter-path" in cmd and "/d/adapter" in cmd
    i = cmd.index("--iters")
    assert cmd[i + 1] == str(spec.lora.iters)
    assert "--batch-size" in cmd and "--num-layers" in cmd


def test_mlx_command_no_q_flag():
    # mlx_lm.lora (v0.31) has no -q flag; verify it is never emitted for 3B models
    cmd = mlx_lora_command(by_key("ministral-3-3b-base"), Path("/d"), Path("/a"))
    assert "-q" not in cmd
