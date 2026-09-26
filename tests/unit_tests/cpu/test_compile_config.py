# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging

import pytest
import torch

from torchtitan.components.loss import cross_entropy_loss, mse_loss
from torchtitan.config import LocalCompileConfig
from torchtitan.distributed.local_compile import apply_local_compile, local_compile


@pytest.fixture(autouse=True)
def reset_local_compile():
    apply_local_compile(None)
    yield
    apply_local_compile(None)


def test_local_compile_config_default() -> None:
    config = LocalCompileConfig()
    assert config.regions == ["gated_rmsnorm", "loss"]


def test_local_compile_config_loss_only() -> None:
    config = LocalCompileConfig(regions=["loss"])
    assert config.regions == ["loss"]


def test_local_compile_config_empty_regions() -> None:
    config = LocalCompileConfig(regions=[])
    assert config.regions == []


def test_apply_local_compile_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match=r"foo.*registered values"):
        apply_local_compile(LocalCompileConfig(regions=["foo"]))


def test_local_compile_binds_once_from_existing_config(monkeypatch, caplog) -> None:
    compiled_calls = []

    @local_compile("test_local_compile", batch_invariant=False)
    def fn(value: int) -> tuple[str, int]:
        return "eager", value

    def fake_compile(reference, **kwargs):
        compiled_calls.append((reference, kwargs))

        def compiled(value: int) -> tuple[str, int]:
            return "compiled", value

        return compiled

    monkeypatch.setattr(torch, "compile", fake_compile)
    apply_local_compile(LocalCompileConfig(regions=["test_local_compile"]))

    assert fn(3) == ("compiled", 3)
    assert len(compiled_calls) == 1
    assert compiled_calls[0][1]["fullgraph"] is True

    monkeypatch.setattr(
        "torchtitan.distributed.local_compile.is_in_batch_invariant_mode",
        lambda: True,
    )
    with caplog.at_level(logging.WARNING):
        apply_local_compile(LocalCompileConfig(regions=["test_local_compile"]))
    assert fn(3) == ("eager", 3)
    assert "test_local_compile" in caplog.text


def test_loss_functions_use_local_compile(monkeypatch) -> None:
    compiled_names = []

    def fake_compile(reference, **kwargs):
        del kwargs
        compiled_names.append(reference.__name__)
        return reference

    monkeypatch.setattr(torch, "compile", fake_compile)
    apply_local_compile(LocalCompileConfig(regions=["loss"]))

    assert compiled_names == [cross_entropy_loss.__name__, mse_loss.__name__]
