#!/usr/bin/env python3
import argparse
from safetensors.torch import load_file, save_file


ACTION_PREFIXES = ("action_", "action_blocks.")


def is_action_key(name: str) -> bool:
    return name.startswith(ACTION_PREFIXES)


def crop_action_tensor(name, tensor, new_in_dim, new_out_dim):
    if name.endswith("action_proj_in.weight"):
        if tensor.shape[1] < new_in_dim:
            raise ValueError(f"{name} has in_dim {tensor.shape[1]} < {new_in_dim}")
        return tensor[:, :new_in_dim].contiguous()
    if name.endswith("action_proj_out.weight"):
        if tensor.shape[0] < new_out_dim:
            raise ValueError(f"{name} has out_dim {tensor.shape[0]} < {new_out_dim}")
        return tensor[:new_out_dim, :].contiguous()
    if name.endswith("action_proj_out.bias"):
        if tensor.shape[0] < new_out_dim:
            raise ValueError(f"{name} has out_dim {tensor.shape[0]} < {new_out_dim}")
        return tensor[:new_out_dim].contiguous()
    if name.endswith("action_state"):
        if tensor.shape[-1] < new_in_dim:
            raise ValueError(f"{name} has in_dim {tensor.shape[-1]} < {new_in_dim}")
        return tensor[..., :new_in_dim].contiguous()
    return tensor


def discarded_info(name, tensor, new_in_dim, new_out_dim):
    if name.endswith("action_proj_in.weight") and tensor.shape[1] > new_in_dim:
        return f"discard cols [{new_in_dim}:{tensor.shape[1]}]"
    if name.endswith("action_proj_out.weight") and tensor.shape[0] > new_out_dim:
        return f"discard rows [{new_out_dim}:{tensor.shape[0]}]"
    if name.endswith("action_proj_out.bias") and tensor.shape[0] > new_out_dim:
        return f"discard elems [{new_out_dim}:{tensor.shape[0]}]"
    if name.endswith("action_state") and tensor.shape[-1] > new_in_dim:
        return f"discard last-dim [{new_in_dim}:{tensor.shape[-1]}]"
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Merge video weights into an action ckpt, optionally crop action dims."
    )
    parser.add_argument("--video", required=True, help="Pure video ckpt (.safetensors)")
    parser.add_argument("--action", required=True, help="Action ckpt (.safetensors)")
    parser.add_argument("--out", required=True, help="Output merged ckpt (.safetensors)")
    parser.add_argument("--action-in-dim", type=int, default=None, help="Crop action_in_channels")
    parser.add_argument("--action-out-dim", type=int, default=None, help="Crop action_out_channels")
    parser.add_argument("--print-discarded", action="store_true", help="Print discarded slices")
    parser.add_argument("--dry-run", action="store_true", help="Only print changes, do not save")
    args = parser.parse_args()

    video_state = load_file(args.video)
    action_state = load_file(args.action)

    replaced = []
    skipped = []

    for name, tensor in video_state.items():
        if is_action_key(name):
            continue
        if name in action_state and action_state[name].shape == tensor.shape:
            action_state[name] = tensor
            replaced.append((name, tuple(tensor.shape)))
        else:
            target_shape = tuple(action_state[name].shape) if name in action_state else None
            skipped.append((name, tuple(tensor.shape), target_shape))

    print(f"Video params replaced: {len(replaced)}")
    for name, shape in replaced:
        print(f"  {name}: {shape}")

    if skipped:
        print(f"Video params skipped: {len(skipped)}")
        for name, src_shape, dst_shape in skipped:
            print(f"  {name}: src={src_shape} dst={dst_shape}")

    if args.action_in_dim is not None or args.action_out_dim is not None:
        new_in = args.action_in_dim
        new_out = args.action_out_dim
        if new_in is None or new_out is None:
            raise ValueError("Both --action-in-dim and --action-out-dim are required to crop.")
        cropped = []
        for name, tensor in list(action_state.items()):
            if not is_action_key(name):
                continue
            new_tensor = crop_action_tensor(name, tensor, new_in, new_out)
            if new_tensor.shape != tensor.shape:
                info = discarded_info(name, tensor, new_in, new_out)
                cropped.append((name, tuple(tensor.shape), tuple(new_tensor.shape), info))
            action_state[name] = new_tensor

        if cropped:
            print("Cropped action params:")
            for name, old_shape, new_shape, info in cropped:
                line = f"  {name}: {old_shape} -> {new_shape}"
                if args.print_discarded and info is not None:
                    line += f" | {info} (random init in target model)"
                print(line)

    if args.dry_run:
        return

    save_file(action_state, args.out)
    print(f"Saved merged ckpt: {args.out}")


if __name__ == "__main__":
    main()
