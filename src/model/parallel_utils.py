from __future__ import annotations

import torch
import torch.nn as nn


def parse_devices(
    devices_arg: str | None = None,
    device_arg: str | None = None,
    default_to_all_cuda: bool = True,
) -> list[str]:
    if devices_arg:
        devices = [d.strip() for d in devices_arg.split(",") if d.strip()]
        if not devices:
            raise ValueError("--devices 为空，请传入类似 cuda:0,cuda:1")
        return devices

    if device_arg:
        device_arg = str(device_arg).strip()
        if device_arg == "cuda":
            if torch.cuda.is_available():
                count = torch.cuda.device_count()
                if count > 0 and default_to_all_cuda:
                    return [f"cuda:{i}" for i in range(count)]
                if count > 0:
                    return ["cuda:0"]
            return ["cpu"]
        return [device_arg]

    if torch.cuda.is_available():
        count = torch.cuda.device_count()
        if count > 0 and default_to_all_cuda:
            return [f"cuda:{i}" for i in range(count)]
        if count > 0:
            return ["cuda:0"]

    return ["cpu"]


def primary_device(devices: list[str]) -> str:
    return devices[0] if devices else "cpu"


def use_data_parallel(devices: list[str]) -> bool:
    return len(devices) > 1 and all(str(d).startswith("cuda") for d in devices)


def cuda_device_ids(devices: list[str]) -> list[int]:
    ids = []
    for dev in devices:
        if not str(dev).startswith("cuda"):
            raise ValueError(f"Only CUDA devices support DataParallel, got {dev}")
        if ":" in str(dev):
            ids.append(int(str(dev).split(":", 1)[1]))
        else:
            ids.append(0)
    return ids


def unwrap_model(model: nn.Module) -> nn.Module:
    while isinstance(model, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
        model = model.module
    return model


def maybe_wrap_data_parallel(model: nn.Module, devices: list[str]) -> nn.Module:
    model = unwrap_model(model)
    if not use_data_parallel(devices):
        return model
    device_ids = cuda_device_ids(devices)
    return nn.DataParallel(model, device_ids=device_ids, output_device=device_ids[0])


def _strip_parallel_key(key: str) -> str:
    if key.startswith("module."):
        key = key[len("module."):]
    return key.replace(".module.", ".")


def sanitize_state_dict_keys(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {_strip_parallel_key(k): v for k, v in state_dict.items()}


def load_state_dict_flexible(model: nn.Module, state_dict: dict[str, torch.Tensor], strict: bool = True) -> None:
    base = unwrap_model(model)
    try:
        base.load_state_dict(state_dict, strict=strict)
        return
    except RuntimeError:
        pass

    stripped = sanitize_state_dict_keys(state_dict)
    base.load_state_dict(stripped, strict=strict)
