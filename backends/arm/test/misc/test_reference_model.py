# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from types import SimpleNamespace

import pytest
import torch
from executorch.backends.arm.test import runner_utils
from executorch.backends.arm.tosa import reference_model


def test_dispatch_runs_only_tosa_backend_delegate(monkeypatch) -> None:
    dispatch = reference_model.TosaReferenceModelDispatch()
    lowered = SimpleNamespace(backend_id="TOSABackend")
    monkeypatch.setattr(
        dispatch, "_tosa_dispatch", lambda module, inputs: (module, inputs)
    )

    result = dispatch.__torch_function__(
        torch._higher_order_ops.executorch_call_delegate,
        (),
        (lowered, "input"),
    )

    assert result == (lowered, ("input",))
    assert dispatch.ran_tosa_dispatch


def test_runner_utils_preserves_reference_model_imports() -> None:
    assert (
        runner_utils.TosaReferenceModelDispatch
        is reference_model.TosaReferenceModelDispatch
    )
    assert runner_utils.run_tosa_graph is reference_model.run_tosa_graph
    assert runner_utils.numpy_to_torch_tensor is reference_model.numpy_to_torch_tensor


def test_dispatch_rejects_wrong_backend() -> None:
    dispatch = reference_model.TosaReferenceModelDispatch()
    lowered = SimpleNamespace(backend_id="OtherBackend")

    with pytest.raises(RuntimeError, match="backend_id='OtherBackend'"):
        dispatch.__torch_function__(
            torch._higher_order_ops.executorch_call_delegate,
            (),
            (lowered,),
        )


def test_dispatch_requires_a_tosa_delegate() -> None:
    with pytest.raises(
        RuntimeError,
        match="never ran TOSABackend delegate",
    ):
        with reference_model.TosaReferenceModelDispatch():
            pass

