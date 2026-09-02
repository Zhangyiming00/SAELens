from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

import torch

from sae_lens.saes.sae import TrainingSAE, TrainStepInput, TrainStepOutput
from sae_lens.saes.topk_sae import TopKTrainingSAE, TopKTPWavefrontState


def _sanitize_module_key(hook_name: str) -> str:
    key = "".join(ch if ch.isalnum() else "_" for ch in hook_name).strip("_")
    return key or "hook"


class MultiHookSAE(torch.nn.Module):
    """Single distributed owner for a fixed ordered set of hook SAEs."""

    def __init__(
        self,
        hook_names: list[str],
        sae_by_hook: Mapping[str, torch.nn.Module],
        *,
        enable_tp_wavefront: bool = True,
    ) -> None:
        super().__init__()
        if not hook_names:
            raise ValueError("MultiHookSAE requires at least one hook")
        if len(set(hook_names)) != len(hook_names):
            raise ValueError("MultiHookSAE hook_names must be unique")

        missing = [hook_name for hook_name in hook_names if hook_name not in sae_by_hook]
        if missing:
            raise ValueError(f"Missing SAEs for hooks: {missing}")

        self.hook_names = list(hook_names)
        self.enable_tp_wavefront = enable_tp_wavefront
        self.module_key_by_hook: dict[str, str] = {}
        self.hook_by_module_key: dict[str, str] = {}
        modules: dict[str, torch.nn.Module] = {}
        used_keys: set[str] = set()
        for hook_name in self.hook_names:
            base_key = _sanitize_module_key(hook_name)
            module_key = base_key
            suffix = 1
            while module_key in used_keys:
                module_key = f"{base_key}_{suffix}"
                suffix += 1
            used_keys.add(module_key)
            self.module_key_by_hook[hook_name] = module_key
            self.hook_by_module_key[module_key] = hook_name
            modules[module_key] = sae_by_hook[hook_name]
        self.saes = torch.nn.ModuleDict(modules)

    def _forward_serial(
        self,
        inputs_by_hook: dict[str, TrainStepInput],
    ) -> dict[str, TrainStepOutput]:
        outputs: dict[str, TrainStepOutput] = {}
        for hook_name in self.hook_names:
            module_key = self.module_key_by_hook[hook_name]
            output = self.saes[module_key](inputs_by_hook[hook_name])
            if not isinstance(output, TrainStepOutput):
                raise TypeError(
                    f"SAE for hook {hook_name!r} returned {type(output).__name__}, "
                    "expected TrainStepOutput"
                )
            outputs[hook_name] = output
        return outputs

    def _can_tp_wavefront(self) -> bool:
        if not self.enable_tp_wavefront or len(self.hook_names) < 2:
            return False
        tp_group = None
        for hook_name in self.hook_names:
            sae = self.saes[self.module_key_by_hook[hook_name]]
            if not isinstance(sae, TopKTrainingSAE) or not sae.tp_wavefront_supported():
                return False
            sae_group = sae._tp_group
            if tp_group is None:
                tp_group = sae_group
            elif sae_group is not tp_group:
                # Every hook must issue collectives on the same TP membership
                # in the same order. Different groups cannot safely share the
                # cross-hook wavefront protocol.
                return False
        return True

    def _forward_tp_wavefront(
        self,
        inputs_by_hook: dict[str, TrainStepInput],
    ) -> dict[str, TrainStepOutput]:
        """Overlap adjacent hooks' TP all-gathers and decoder all-reduces."""
        states: dict[str, TopKTPWavefrontState] = {}
        current_hook = self.hook_names[0]
        current_sae = cast(
            TopKTrainingSAE, self.saes[self.module_key_by_hook[current_hook]]
        )
        current_state = current_sae.tp_wavefront_encode_launch(
            inputs_by_hook[current_hook]
        )

        for next_hook in self.hook_names[1:]:
            next_sae = cast(
                TopKTrainingSAE, self.saes[self.module_key_by_hook[next_hook]]
            )
            # AG(current) overlaps this next hook's local encoder work.
            next_state = next_sae.tp_wavefront_encode_launch(inputs_by_hook[next_hook])
            current_sae.tp_wavefront_decode_launch(current_state)
            states[current_hook] = current_state
            current_hook, current_sae, current_state = (
                next_hook,
                next_sae,
                next_state,
            )

        current_sae.tp_wavefront_decode_launch(current_state)
        states[current_hook] = current_state

        return {
            hook_name: cast(
                TopKTrainingSAE,
                self.saes[self.module_key_by_hook[hook_name]],
            ).tp_wavefront_finish(states[hook_name])
            for hook_name in self.hook_names
        }

    def forward(
        self,
        inputs_by_hook: dict[str, TrainStepInput],
    ) -> dict[str, TrainStepOutput]:
        expected_hooks = set(self.hook_names)
        actual_hooks = set(inputs_by_hook)
        if actual_hooks != expected_hooks:
            raise ValueError(
                "MultiHookSAE forward requires exact hook set "
                f"{self.hook_names}; got {sorted(actual_hooks)}"
            )
        if self._can_tp_wavefront():
            return self._forward_tp_wavefront(inputs_by_hook)
        return self._forward_serial(inputs_by_hook)

    def get_raw_sae(self, hook_name: str) -> torch.nn.Module:
        module_key = self.module_key_by_hook[hook_name]
        return self.saes[module_key]

    def split_state_dict_by_hook(
        self,
        root_state_dict: Mapping[str, Any],
        *,
        strict: bool = True,
    ) -> dict[str, dict[str, Any]]:
        split: dict[str, dict[str, Any]] = {hook_name: {} for hook_name in self.hook_names}
        prefixes_by_hook = {
            hook_name: (
                f"saes.{module_key}.",
                f"module.saes.{module_key}.",
            )
            for hook_name, module_key in self.module_key_by_hook.items()
        }

        for key, value in root_state_dict.items():
            for hook_name, prefixes in prefixes_by_hook.items():
                for prefix in prefixes:
                    if key.startswith(prefix):
                        split[hook_name][key.removeprefix(prefix)] = value
                        break
                else:
                    continue
                break

        if strict:
            missing = [hook_name for hook_name, state in split.items() if not state]
            if missing:
                raise ValueError(f"Missing state_dict entries for hooks: {missing}")
        return split

    def merge_state_dict_by_hook(
        self,
        state_dict_by_hook: Mapping[str, Mapping[str, Any]],
    ) -> dict[str, Any]:
        missing = [
            hook_name for hook_name in self.hook_names if hook_name not in state_dict_by_hook
        ]
        if missing:
            raise ValueError(f"Missing state_dict for hooks: {missing}")

        merged: dict[str, Any] = {}
        for hook_name in self.hook_names:
            module_key = self.module_key_by_hook[hook_name]
            prefix = f"saes.{module_key}."
            for key, value in state_dict_by_hook[hook_name].items():
                merged[f"{prefix}{key}"] = value
        return merged

    def raw_sae_by_hook(self) -> dict[str, TrainingSAE[Any]]:
        return {
            hook_name: cast(
                TrainingSAE[Any],
                self.saes[self.module_key_by_hook[hook_name]],
            )
            for hook_name in self.hook_names
        }
