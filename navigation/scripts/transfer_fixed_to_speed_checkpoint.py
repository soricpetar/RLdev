from __future__ import annotations

import argparse
from collections import OrderedDict
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Expand a 5-action fixed-speed FieldNav checkpoint into a 15-action speed-control checkpoint without changing hidden/model size."
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--preferred-throttle-idx",
        type=int,
        default=2,
        choices=[0, 1, 2],
        help="Throttle slot to receive the original fixed-speed steering logits: 0=brake, 1=coast, 2=accelerate.",
    )
    parser.add_argument(
        "--nonpreferred-bias-offset",
        type=float,
        default=-6.0,
        help="Bias penalty for non-preferred throttle rows so argmax initially mimics the fixed-speed policy.",
    )
    args = parser.parse_args()

    state = torch.load(args.source, map_location="cpu", weights_only=False)
    if not isinstance(state, dict):
        raise TypeError(f"Expected state dict, got {type(state)!r}")
    if tuple(state["decoder.decoder.weight"].shape) != (5, 256):
        raise ValueError(f"Expected 5x256 fixed-speed decoder, got {tuple(state['decoder.decoder.weight'].shape)}")

    out_state = OrderedDict((k, v.clone() if torch.is_tensor(v) else v) for k, v in state.items())
    fixed_w = state["decoder.decoder.weight"]
    fixed_b = state["decoder.decoder.bias"]
    speed_w = fixed_w.new_empty((15, fixed_w.shape[1]))
    speed_b = fixed_b.new_empty((15,))

    # FieldNav speed action layout: action = steering_idx * 3 + throttle_idx,
    # throttle_idx 0=brake, 1=coast, 2=accelerate.
    for steering_idx in range(5):
        for throttle_idx in range(3):
            row = steering_idx * 3 + throttle_idx
            speed_w[row] = fixed_w[steering_idx]
            speed_b[row] = fixed_b[steering_idx]
            if throttle_idx != args.preferred_throttle_idx:
                speed_b[row] += args.nonpreferred_bias_offset

    out_state["decoder.decoder.weight"] = speed_w
    out_state["decoder.decoder.bias"] = speed_b
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out_state, args.out)
    print(f"wrote {args.out}")
    print(f"params unchanged except decoder action rows: source_rows=5 speed_rows=15 hidden={fixed_w.shape[1]}")


if __name__ == "__main__":
    main()
