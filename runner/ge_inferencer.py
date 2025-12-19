import os, random, math
from pathlib import Path
from typing import Any, Dict, List

from datetime import datetime, timedelta
import argparse
import json
import importlib
# ----------------------------------------------------
import matplotlib.pyplot as plt
import matplotlib
from yaml import load, dump, Loader, Dumper
import numpy as np
from tqdm import tqdm
import torch
from torch import distributed as dist
from einops import rearrange
from copy import deepcopy
import transformers
import logging

# ----------------------------------------------------
import diffusers
from diffusers.optimization import get_scheduler
from diffusers.training_utils import (
    cast_training_params,
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
)

# ----------------------------------------------------
from utils.model_utils import load_condition_models, load_latent_models, load_vae_models, load_diffusion_model, count_model_parameters, unwrap_model

# ----------------------------------------------------
from torch.utils.tensorboard import SummaryWriter
from utils import init_logging, import_custom_class, save_video
from utils.data_utils import get_latents, get_text_conditions, gen_noise_from_condition_frame_latent, randn_tensor, apply_color_jitter_to_video
from data.utils.statistics import StatisticInfo
from PIL import Image


def save_image(tensor, path):
    """
    Save a torch tensor as an image.
    Args:
        tensor: torch.Tensor of shape (b, c, t, h, w) or (b, c, h, w)
        path: str, path to save the image
    """
    # Take first batch and first time frame if applicable
    if tensor.dim() == 5:  # (b, c, t, h, w)
        img = tensor[0, :, 0, :, :]  # Take first batch, first frame
    elif tensor.dim() == 4:  # (b, c, h, w)
        img = tensor[0]  # Take first batch
    else:
        img = tensor
    
    # Convert to numpy and transpose to (h, w, c)
    img = img.detach().cpu().float()
    
    # Denormalize if needed (assuming normalized to [-1, 1] or [0, 1])
    if img.min() < 0:
        img = (img + 1.0) / 2.0  # [-1, 1] -> [0, 1]
    
    img = torch.clamp(img, 0, 1)
    img = (img * 255).numpy().astype(np.uint8)
    
    # Transpose from (c, h, w) to (h, w, c)
    if img.shape[0] in [1, 3, 4]:  # Channel dimension is first
        img = np.transpose(img, (1, 2, 0))
    
    # Handle grayscale
    if img.shape[-1] == 1:
        img = img.squeeze(-1)
    
    # Save using PIL
    Image.fromarray(img).save(path)



class Inferencer:

    def __init__(self, config_file, output_dir=None, weight_dtype=torch.bfloat16, device="cuda:0", action_norm_type="meanstd") -> None:
        
        cd = load(open(config_file, "r"), Loader=Loader)
        args = argparse.Namespace(**cd)
        args.lr = float(args.lr)
        args.epsilon = float(args.epsilon)
        args.weight_decay = float(args.weight_decay)

        self.args = args

        if output_dir is not None:
            self.args.output_dir = output_dir

        if self.args.load_weights == False:
            print('You are not loading the pretrained weights, please check the code.')

        # Tokenizers
        self.tokenizer = None

        # Text encoders
        self.text_encoder = None

        # Denoisers
        self.diffusion_model = None
        self.unet = None

        # Autoencoders
        self.vae = None

        # Scheduler
        self.scheduler = None

        self.args.output_dir = Path(self.args.output_dir)
        self.args.output_dir.mkdir(parents=True, exist_ok=True)

        current_time = datetime.now()
        start_time = current_time.strftime("%Y_%m_%d_%H_%M_%S")
        self.save_folder = os.path.join(self.args.output_dir, start_time)
        if getattr(self.args, "sub_folder", False):
            self.save_folder = os.path.join(self.args.output_dir, self.args.sub_folder)
        os.makedirs(self.save_folder, exist_ok=True)

        self._invalid_occlusion_logged = False

        args_dict = vars(deepcopy(self.args))
        for k, v in args_dict.items():
            args_dict[k] = str(v)
        with open(os.path.join(self.save_folder, 'config.json'), "w") as file:
            json.dump(args_dict, file, indent=4, sort_keys=False)
        
        self.weight_dtype = weight_dtype
        self.device = device


        self.StatisticInfo = StatisticInfo
        if self.args.data['val'].get('stat_file', None) is not None:
            with open(self.args.data['val']['stat_file'], "r") as f:
                self.StatisticInfo = json.load(f)

        self.action_norm_type = action_norm_type

    def prepare_val_dataset(self) -> None:
        if not hasattr(self.args, "val_data_class"):
            self.args.val_data_class = self.args.train_data_class
        print(f"Validation Dataset: {self.args.val_data_class}")

        val_dataset_class = import_custom_class(
            self.args.val_data_class, self.args.val_data_class_path
        )
        
        self.args.data['val'].update({"fix_epiidx": 0, "fix_sidx":0, "fix_mem_idx":[0,0,0,0]})

        self.val_dataset = val_dataset_class(**self.args.data['val'])
        
        # Custom collate_fn to handle string captions correctly
        def collate_fn(batch):
            """Custom collate function that preserves string captions"""
            # Get keys from first sample
            keys = batch[0].keys()
            collated = {}
            for key in keys:
                values = [item[key] for item in batch]
                # For caption, keep as list of strings
                if key == 'caption':
                    collated[key] = [str(v) if not isinstance(v, str) else v for v in values]
                # For other fields, use default collate behavior
                else:
                    try:
                        collated[key] = torch.utils.data.default_collate(values)
                    except (TypeError, RuntimeError):
                        # If default collate fails, keep as list
                        collated[key] = values
            return collated
        
        self.val_dataloader = torch.utils.data.DataLoader(
            self.val_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn, num_workers=0
        )


    def prepare_models(self,):

        print("Initializing models")
        device = self.device
        dtype = self.weight_dtype

        ### Load Tokenizer
        tokenizer_class = import_custom_class(
            self.args.tokenizer_class, getattr(self.args, "tokenizer_class_path", "transformers")
        )
        textenc_class = import_custom_class(
            self.args.textenc_class, getattr(self.args, "textenc_class_path", "transformers")
        )
        cond_models = load_condition_models(
            tokenizer_class, textenc_class,
            self.args.pretrained_model_name_or_path if not hasattr(self.args, "tokenizer_pretrained_model_name_or_path") else self.args.tokenizer_pretrained_model_name_or_path,
            load_weights=self.args.load_weights
        )
        self.tokenizer, text_encoder = cond_models["tokenizer"], cond_models["text_encoder"]
        self.text_encoder = text_encoder.to(device, dtype=dtype).eval()
        self.text_uncond = get_text_conditions(self.tokenizer, self.text_encoder, prompt="")
        self.uncond_prompt_embeds = self.text_uncond['prompt_embeds']
        self.uncond_prompt_attention_mask = self.text_uncond['prompt_attention_mask']

        ### Load VAE
        vae_class = import_custom_class(
            self.args.vae_class, getattr(self.args, "vae_class_path", "transformers")
        )
        if getattr(self.args, 'vae_path', False):
            self.vae = load_vae_models(vae_class, self.args.vae_path).to(device, dtype=dtype).eval()
        else:
            self.vae = load_latent_models(vae_class, self.args.pretrained_model_name_or_path)["vae"].to(device, dtype=dtype).eval()
        if isinstance(self.vae.latents_mean, List):
            self.vae.latents_mean = torch.FloatTensor(self.vae.latents_mean)
        if isinstance(self.vae.latents_std, List):
            self.vae.latents_std = torch.FloatTensor(self.vae.latents_std)
        if self.vae is not None:
            if self.args.enable_slicing:
                self.vae.enable_slicing()
            if self.args.enable_tiling:
                self.vae.enable_tiling()
        self.SPATIAL_DOWN_RATIO = self.vae.spatial_compression_ratio
        self.TEMPORAL_DOWN_RATIO = self.vae.temporal_compression_ratio
        print(f'SPATIAL_DOWN_RATIO of VAE :{self.SPATIAL_DOWN_RATIO}')
        print(f'TEMPORAL_DOWN_RATIO of VAE :{self.TEMPORAL_DOWN_RATIO}')


        ### Load Diffusion Model
        diffusion_model_class = import_custom_class(
            self.args.diffusion_model_class, getattr(self.args, "diffusion_model_class_path", "transformers")
        )
        self.diffusion_model = load_diffusion_model(
            model_cls=diffusion_model_class,
            model_dir=self.args.diffusion_model['model_path'],
            load_weights=self.args.load_weights and getattr(self.args, "load_diffusion_model_weights", True),
            **self.args.diffusion_model['config']
        ).to(device, dtype=dtype)
        total_params = count_model_parameters(self.diffusion_model)
        print(f'Total parameters for transformer model:{total_params}')


        ### Load Diffuser Scheduler
        diffusion_scheduler_class = import_custom_class(
            self.args.diffusion_scheduler_class, getattr(self.args, "diffusion_scheduler_class_path", "diffusers")
        )
        if hasattr(self.args, "diffusion_scheduler_args"):
            self.scheduler = diffusion_scheduler_class(**self.args.diffusion_scheduler_args)
        else:
            self.scheduler = diffusion_scheduler_class()

        ### Import Inference Pipeline Class
        self.pipeline_class = import_custom_class(
            self.args.pipeline_class, getattr(self.args, "pipeline_class_path", "diffusers")
        )

    def _denormalize_actions(self, batch, preds_tensor, domain_name, action_type, action_space, statistics_domain=None):
        gt_actions = batch['actions'][:, self.args.data['train']['n_previous']:]

        pd_actions_arr = preds_tensor.detach().cpu().to(torch.float32).numpy()
        gt_actions_arr = gt_actions[0].detach().cpu().to(torch.float32).numpy()

        n_dim = pd_actions_arr.shape[-1]
        gripper_dim = 1
        arm_dim = (n_dim - gripper_dim) // 2

        stats_lookup_domain = statistics_domain if statistics_domain else domain_name
        act_stats_key = f"{stats_lookup_domain}_{action_space}"
        if act_stats_key not in self.StatisticInfo:
            raise KeyError(f"Statistics key '{act_stats_key}' not found in StatisticInfo.")
        act_mean = np.expand_dims(np.array(self.StatisticInfo[act_stats_key]["mean"]), axis=0)
        act_std = np.expand_dims(np.array(self.StatisticInfo[act_stats_key]["std"]), axis=0)

        abs_norm_mode = None
        if hasattr(self.args, "data"):
            if "train" in self.args.data:
                abs_norm_mode = self.args.data["train"].get("action_abs_norm")
            if abs_norm_mode is None and "val" in self.args.data:
                abs_norm_mode = self.args.data["val"].get("action_abs_norm")
        abs_norm_mode = (abs_norm_mode or "minmax").lower()

        dataset_name = None
        data_root = None
        try:
            val_cfg = self.args.data.get("val", {})
            force_stats = val_cfg.get("force_dataset_statistics_consistency")
            if force_stats:
                dataset_name = force_stats
            else:
                val_roots = val_cfg.get("data_roots")
                if isinstance(val_roots, (list, tuple)) and len(val_roots) > 0:
                    data_root = val_roots[0]
                elif isinstance(val_roots, str):
                    data_root = val_roots
                if data_root:
                    dataset_name = os.path.basename(data_root.rstrip('/'))
        except Exception:
            dataset_name = None
            data_root = None

        if action_type == "absolute":
            candidates = [f"{stats_lookup_domain}_{action_space}"]
            if dataset_name:
                candidates.append(f"{dataset_name}_joint_action_{action_space}")
                candidates.append(f"{dataset_name}_{action_space}")
            candidates.append(f"calvin_{action_space}")

            def _select_stats(fields):
                for key in candidates:
                    entry = self.StatisticInfo.get(key)
                    if entry is None:
                        continue
                    if all(field in entry for field in fields):
                        return key, entry
                return None, None

            preferred_fields = ["min", "max"] if abs_norm_mode == "minmax" else ["mean", "std"]
            chosen_key, chosen_entry = _select_stats(preferred_fields)

            if chosen_key is None:
                fallback_mode = "zscore" if abs_norm_mode == "minmax" else "minmax"
                fallback_fields = ["min", "max"] if fallback_mode == "minmax" else ["mean", "std"]
                chosen_key, chosen_entry = _select_stats(fallback_fields)
                if chosen_key:
                    print(f"⚠️  Requested abs_norm_mode='{abs_norm_mode}' but statistics missing for keys {candidates}. "
                          f"Falling back to '{fallback_mode}'.")
                    abs_norm_mode = fallback_mode
                else:
                    raise KeyError(f"No suitable statistics found for keys {candidates} "
                                   f"(looked for fields {preferred_fields}).")

            if abs_norm_mode == "minmax":
                act_min = np.expand_dims(np.array(chosen_entry["min"]), axis=0)
                act_max = np.expand_dims(np.array(chosen_entry["max"]), axis=0)
                pd_actions_arr = ((pd_actions_arr + 1.0) / 2.0) * (act_max - act_min) + act_min
                gt_actions_arr = ((gt_actions_arr + 1.0) / 2.0) * (act_max - act_min) + act_min
            else:
                act_mean_sel = np.expand_dims(np.array(chosen_entry["mean"]), axis=0)
                act_std_sel = np.expand_dims(np.array(chosen_entry["std"]), axis=0)
                pd_actions_arr = pd_actions_arr * act_std_sel + act_mean_sel
                gt_actions_arr = gt_actions_arr * act_std_sel + act_mean_sel

        elif action_type == "delta":
            state = batch["state"][0].data.cpu().float().numpy()
            state = state * act_std + act_mean
            delta_stats_key = f"{stats_lookup_domain}_delta_{action_space}"
            dact_mean = np.expand_dims(np.array(self.StatisticInfo[delta_stats_key]["mean"]), axis=0)
            dact_std = np.expand_dims(np.array(self.StatisticInfo[delta_stats_key]["std"]), axis=0)
            pd_actions_arr = pd_actions_arr * dact_std + dact_mean
            gt_actions_arr = gt_actions_arr * dact_std + dact_mean
            pd_actions_arr[:, :arm_dim] = np.cumsum(pd_actions_arr[:, :arm_dim], axis=0) + state[:, :arm_dim]
            gt_actions_arr[:, :arm_dim] = np.cumsum(gt_actions_arr[:, :arm_dim], axis=0) + state[:, :arm_dim]
            pd_actions_arr[:, arm_dim + gripper_dim:2 * arm_dim + gripper_dim] = (
                np.cumsum(pd_actions_arr[:, arm_dim + gripper_dim:2 * arm_dim + gripper_dim], axis=0)
                + state[:, arm_dim + gripper_dim:2 * arm_dim + gripper_dim]
            )
            gt_actions_arr[:, arm_dim + gripper_dim:2 * arm_dim + gripper_dim] = (
                np.cumsum(gt_actions_arr[:, arm_dim + gripper_dim:2 * arm_dim + gripper_dim], axis=0)
                + state[:, arm_dim + gripper_dim:2 * arm_dim + gripper_dim]
            )

        elif action_type == "relative":
            state = batch["state"][0].data.cpu().float().numpy()
            pd_actions_arr[:, :arm_dim] = pd_actions_arr[:, :arm_dim] + state[:, :arm_dim]
            pd_actions_arr[:, arm_dim + gripper_dim:2 * arm_dim + gripper_dim] = (
                pd_actions_arr[:, arm_dim + gripper_dim:2 * arm_dim + gripper_dim] + state[:, arm_dim + gripper_dim:2 * arm_dim + gripper_dim]
            )
            pd_actions_arr = pd_actions_arr * act_std + act_mean
            gt_actions_arr[:, :arm_dim] = gt_actions_arr[:, :arm_dim] + state[:, :arm_dim]
            gt_actions_arr[:, arm_dim + gripper_dim:2 * arm_dim + gripper_dim] = (
                gt_actions_arr[:, arm_dim + gripper_dim:2 * arm_dim + gripper_dim] + state[:, arm_dim + gripper_dim:2 * arm_dim + gripper_dim]
            )
            gt_actions_arr = gt_actions_arr * act_std + act_mean

        else:
            raise NotImplementedError(f"Unsupported action_type {action_type}")

        return pd_actions_arr, gt_actions_arr

    def _plot_rollout(self, info, save_plot_path=None):
        if save_plot_path is not None:
            matplotlib.use("Agg")

        action_dim = info["action_dim"]
        gt_action_across_time = info["gt_action_across_time"]
        pred_action_across_time = info["pred_action_across_time"]
        inference_points = info["inference_points"]
        traj_id = info["traj_id"]
        mse = info["mse"]
        modality_desc = info.get("modality_desc", "")

        fig, axes = plt.subplots(action_dim, 1, figsize=(10, max(4 * action_dim + 2, 6)))
        if action_dim == 1:
            axes = [axes]

        plt.subplots_adjust(top=0.92, left=0.1, right=0.96, hspace=0.4)
        title_text = (
            f"Trajectory Analysis - ID: {traj_id}\n"
            f"Modalities: {modality_desc}\n"
            f"Unnormalized MSE: {mse:.6f}"
        )
        fig.suptitle(title_text, fontsize=14, fontweight="bold", color="#2E86AB", y=0.95)

        steps = gt_action_across_time.shape[0]

        occlusion_marks = info.get("occlusion_marks") or []
        occlusion_label = info.get("occlusion_label")
        valid_occlusion_marks = []
        for mark in occlusion_marks:
            try:
                mark_int = int(mark)
            except (TypeError, ValueError):
                continue
            if 0 <= mark_int < steps:
                valid_occlusion_marks.append(mark_int)
        valid_occlusion_marks = sorted(set(valid_occlusion_marks))

        for i, ax in enumerate(axes):
            ax.plot(gt_action_across_time[:, i], label="gt action", linewidth=2, color="#2A6F97")
            ax.plot(pred_action_across_time[:, i], label="pred action", linewidth=2, color="#E07A5F")
            for point_idx, step in enumerate(inference_points):
                if step >= steps:
                    continue
                marker_size = 6 if point_idx == 0 else 4
                ax.plot(step, pred_action_across_time[step, i], "ro", markersize=marker_size, label="inference point" if point_idx == 0 else "")
            for idx, mark in enumerate(valid_occlusion_marks):
                label = None
                if idx == 0 and i == 0:
                    label = "Occlusion start"
                    if occlusion_label:
                        label += f" ({occlusion_label})"
                ax.axvline(mark, color="#FFBF00", linestyle="--", linewidth=1.4, label=label)
            ax.set_title(f"Action Dimension {i}", fontsize=12, fontweight="bold", pad=10)
            ax.set_xlabel("Time Step", fontsize=10)
            ax.set_ylabel("Value", fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.legend(loc="upper right", framealpha=0.9)

        if save_plot_path:
            plt.savefig(save_plot_path, dpi=300, bbox_inches="tight")
            plt.close(fig)
        else:
            plt.show()

    def _apply_occlusion(self, dataset, batch):
        occlude_view = getattr(self.args, "occlude_view", None)
        if not occlude_view:
            return batch

        video = batch.get("video")
        if video is None:
            return batch

        # Clear previous markers to avoid stale values when occlusion is disabled.
        batch.pop("_occlusion_start", None)
        batch.pop("_occlusion_end", None)
        batch.pop("_occlusion_view_idx", None)
        batch.pop("_occlusion_view", None)

        view_idx = None
        if isinstance(occlude_view, str):
            valid_cam = getattr(dataset, "valid_cam", None)
            if valid_cam and occlude_view in valid_cam:
                view_idx = valid_cam.index(occlude_view)
            elif occlude_view.isdigit():
                view_idx = int(occlude_view)
        else:
            try:
                view_idx = int(occlude_view)
            except (TypeError, ValueError):
                view_idx = None

        if view_idx is None:
            if not self._invalid_occlusion_logged:
                print(f"⚠️  Occlusion view '{occlude_view}' not found. Available cameras: {getattr(dataset, 'valid_cam', None)}.")
                self._invalid_occlusion_logged = True
            return batch

        total_views = video.shape[2]
        if view_idx < 0 or view_idx >= total_views:
            if not self._invalid_occlusion_logged:
                print(f"⚠️  Occlusion view index {view_idx} out of range (total views={total_views}).")
                self._invalid_occlusion_logged = True
            return batch

        total_time = video.shape[3]

        # Build a mapping from local frame index -> global timestep so we can
        # apply the occlusion window even when it falls outside the clipped clip.
        mem_indices = getattr(dataset, "fix_mem_idx", None) or []
        try:
            mem_indices = [int(v) for v in mem_indices]
        except TypeError:
            mem_indices = []

        chunk_start = getattr(dataset, "fix_sidx", None)
        chunk_len = getattr(dataset, "action_chunk", None)
        stride = getattr(dataset, "video_temporal_stride", 1)
        if stride is None or stride <= 0:
            stride = 1

        global_indices = list(mem_indices)
        if chunk_start is not None and chunk_len is not None:
            chunk_start = int(chunk_start)
            chunk_len = int(chunk_len)
            chunk_range = list(range(chunk_start, chunk_start + chunk_len))
            if stride > 1:
                chunk_range = chunk_range[stride - 1 :: stride]
            global_indices.extend(chunk_range)

        if not global_indices:
            global_indices = list(range(total_time))

        if len(global_indices) < total_time:
            last_val = global_indices[-1] if global_indices else 0
            while len(global_indices) < total_time:
                last_val += stride
                global_indices.append(last_val)
        elif len(global_indices) > total_time:
            global_indices = global_indices[:total_time]

        start_global = getattr(self.args, "occlude_start", None)
        end_global = getattr(self.args, "occlude_end", None)

        if start_global is None:
            start_global = global_indices[0]
        if end_global is not None and end_global <= start_global:
            if not self._invalid_occlusion_logged:
                print(f"⚠️  Occlusion end ({end_global}) <= start ({start_global}); skipping occlusion.")
                self._invalid_occlusion_logged = True
            return batch

        if end_global is None:
            indices_to_zero = [i for i, g in enumerate(global_indices) if g >= start_global]
        else:
            indices_to_zero = [i for i, g in enumerate(global_indices) if start_global <= g < end_global]

        if not indices_to_zero:
            return batch

        rel_start = indices_to_zero[0]
        rel_end = indices_to_zero[-1] + 1
        try:
            print(
                f"[DEBUG openloop] Applied occlusion on view={occlude_view} (idx={view_idx}); "
                f"global=[{start_global},{end_global}) -> local=[{rel_start},{rel_end}); "
                f"chunk_start={chunk_start}, chunk_len={chunk_len}, stride={stride}, "
                f"clip_len={total_time}"
            )
        except Exception:
            pass

        if isinstance(video, torch.Tensor):
            video[:, :, view_idx, rel_start:rel_end, ...] = 0.0
        else:
            video[:, :, view_idx, rel_start:rel_end, ...] = 0.0

        batch["_occlusion_start"] = int(start_global)
        batch["_occlusion_end"] = int(end_global) if end_global is not None else None
        valid_cam = getattr(dataset, "valid_cam", None)
        if valid_cam and 0 <= view_idx < len(valid_cam):
            batch["_occlusion_view"] = valid_cam[view_idx]
        else:
            batch["_occlusion_view"] = str(occlude_view)
        batch["_occlusion_view_idx"] = int(view_idx)
        return batch

    def validate(
        self,
        model_save_dir,
        global_step,
        n_view=1,
        n_chunk_video=1,
        n_chunk_action=10,
        n_validation=1,
        domain_name="agibotworld",
        tasks_per_run=None,
        episodes_per_task=1,
        statistics_domain=None,
    ):

        os.makedirs(model_save_dir,exist_ok=True)

        pipe = self.pipeline_class(
            self.scheduler, self.vae, self.text_encoder, self.tokenizer, self.diffusion_model
        )

        assert(self.args.return_action | self.args.return_video)


        if self.args.return_action:
            n_chunk_video = 1
            action_type = self.args.data["train"]["action_type"]
            action_space = self.args.data["train"]["action_space"]
            

        if self.args.return_video:
            n_chunk_action = 1


        dataset = self.val_dataloader.dataset
        task_spans = getattr(dataset, "task_spans", [])
        total_tasks_available = len(task_spans)

        episodes_per_task = max(1, int(episodes_per_task))
        if total_tasks_available == 0:
            tasks_per_run_effective = 1
        else:
            if tasks_per_run is None or tasks_per_run <= 0:
                tasks_per_run_effective = math.ceil(n_validation / episodes_per_task)
            else:
                tasks_per_run_effective = tasks_per_run
            tasks_per_run_effective = max(1, min(tasks_per_run_effective, total_tasks_available))

        max_iterations = (
            min(n_validation, tasks_per_run_effective * episodes_per_task)
            if total_tasks_available
            else n_validation
        )

        print(
            f"[DEBUG validate] total_tasks_available={total_tasks_available}, "
            f"tasks_per_run={tasks_per_run_effective}, "
            f"episodes_per_task={episodes_per_task}, "
            f"max_iterations={max_iterations}"
        )

        occlusion_label = getattr(self.args, "occlude_view", None)
        for i_validation in range(max_iterations):
            chunk_size = self.args.data["train"]["action_chunk"]
            n_prev = self.args.data["train"]["n_previous"]
            max_action_steps = n_chunk_action * chunk_size
            occlusion_marks = []
            if total_tasks_available:
                task_idx = i_validation // episodes_per_task
                episode_offset = i_validation % episodes_per_task
                span = task_spans[task_idx]
                span_end = span["start"] + span["length"] - 1
                target_idx = min(span["start"] + episode_offset, span_end)
                dataset.fix_epiidx = target_idx
                current_task = span["task"]
            else:
                dataset.fix_epiidx = i_validation
                current_task = None
            task_name = current_task

            print(
                f"[DEBUG] Processing episode {i_validation}/{max_iterations-1}"
                + (f" (task={current_task})" if current_task else "")
            )
            print(f"[DEBUG] Occlusion config: start={getattr(self.args, 'occlude_start', None)}, end={getattr(self.args, 'occlude_end', None)}, view={occlusion_label}")

            if self.args.return_action:
                self.val_dataloader.dataset.fix_sidx = 0
                self.val_dataloader.dataset.fix_mem_idx = [1 for _ in range(self.args.data['train']['n_previous'])]

                pd_actions_arr_all = None
                gt_actions_arr_all = None

            if self.args.return_action:
                ### openloop result visualization
                fig, axes = plt.subplots(10, 2, figsize=(20, 28), sharex=True)
                axes = axes.flatten()
                total_steps = max_action_steps

            for i_chunk_action in range(n_chunk_action):

                batch = next(iter(self.val_dataloader))
                batch = self._apply_occlusion(self.val_dataloader.dataset, batch)

                occl_start = batch.get("_occlusion_start")
                occl_end = batch.get("_occlusion_end")
                if occl_start is not None:
                    occl_start = int(occl_start)
                    occl_end = int(occl_end) if occl_end is not None else None
                    chunk_start_idx = int(self.val_dataloader.dataset.fix_sidx)
                    chunk_end_idx = chunk_start_idx + chunk_size
                    mark = None
                    # Occlusion already active before this chunk but still ongoing.
                    if occl_start < chunk_start_idx and (occl_end is None or occl_end > chunk_start_idx):
                        mark = i_chunk_action * chunk_size
                    # Occlusion begins inside this chunk.
                    elif chunk_start_idx <= occl_start < chunk_end_idx:
                        mark = i_chunk_action * chunk_size + (occl_start - chunk_start_idx)
                    if (
                        mark is not None
                        and 0 <= mark < max_action_steps
                        and mark not in occlusion_marks
                    ):
                        occlusion_marks.append(mark)
                image = batch['video'][:,:,:,:self.args.data['train']['n_previous']]  # shape b,c,v,t,h,w 
                prompt = batch['caption']
                
                # Debug: print prompt type and value
                print(f"[DEBUG] Prompt type: {type(prompt)}, value: {prompt}")
                # Ensure prompt is a list of strings
                if isinstance(prompt, (list, tuple)):
                    prompt = [str(p) if not isinstance(p, str) else p for p in prompt]
                else:
                    prompt = [str(prompt)]
                print(f"[DEBUG] Prompt after sanitization: {prompt}")
                
                gt_video = batch['video']

                b, c, v, t, h, w = image.shape

                negative_prompt = ''

                batch_size = 1

                image = image[:batch_size]

                image = rearrange(image, 'b c v t h w -> (b v) c t h w')

                if getattr(self.args, "add_state", False):
                    history_action_state = batch["state"][:batch_size] 
                    if history_action_state.shape[1] > 1:
                        history_action_state = history_action_state[:, self.args.data['train']['n_previous']-1:self.args.data['train']['n_previous'], :]
                    history_action_state = history_action_state.contiguous() ### B, 1, C
                else:
                    history_action_state = None

                preds = pipe.infer(
                    image=image,
                    prompt=prompt[:batch_size],
                    negative_prompt=negative_prompt,
                    num_inference_steps=self.args.num_inference_step,
                    decode_timestep=0.03,
                    decode_noise_scale=0.025,
                    guidance_scale=1.0,
                    height=h,
                    width=w,
                    n_view=v,
                    return_action=self.args.return_action,
                    n_prev=self.args.data['train']['n_previous'],
                    chunk=(self.args.data['train']['chunk']-1)//self.TEMPORAL_DOWN_RATIO+1,
                    return_video=self.args.return_video,
                    noise_seed=42,
                    action_chunk=self.args.data['train']['action_chunk'],
                    history_action_state = history_action_state,
                    pixel_wise_timestep = self.args.pixel_wise_timestep,
                    n_chunk=n_chunk_video,
                    action_dim=self.args.diffusion_model["config"]["action_in_channels"],
                )[0]

                save_cap = f'Validation_{i_validation}'
                if task_name:
                    safe_task = "".join(c if c.isalnum() or c in ("_", "-") else "_" for c in task_name)
                    save_cap = f'{save_cap}_{safe_task}'
                task_name = prompt[0] if isinstance(prompt, list) and len(prompt) > 0 else f"episode_{i_validation}"
                print(f"[DEBUG] Saving results for episode {i_validation}, task: {task_name}")

                if self.args.return_video:
                    
                    video = preds['video'].data.cpu()
                    gt_path = os.path.join(model_save_dir, f'{save_cap}_gt.mp4')
                    pred_path = os.path.join(model_save_dir, f'{save_cap}.mp4')

                    save_video(rearrange(gt_video[0].data.cpu(), 'c v t h w -> c t h (v w)', v=n_view), gt_path, fps=(self.args.data['train']['chunk']-1)//self.TEMPORAL_DOWN_RATIO+1)
                    save_video(rearrange(video, '(b v) c t h w -> b c t h (v w)', v=n_view)[0], pred_path, fps=(self.args.data['train']['chunk']-1)//self.TEMPORAL_DOWN_RATIO+1)

                    print(f"[DEBUG] Saved video: {pred_path}")


                if self.args.return_action:
                    pd_actions_arr, gt_actions_arr = self._denormalize_actions(
                        batch, preds['action'][0], domain_name, action_type, action_space, statistics_domain
                    )

                    # n_dim = pd_actions_arr.shape[-1]
                    
                    if pd_actions_arr_all is None:
                        pd_actions_arr_all = pd_actions_arr
                    else:
                        pd_actions_arr_all = np.concatenate((pd_actions_arr_all, pd_actions_arr), axis=0)
                    
                    if gt_actions_arr_all is None:
                        gt_actions_arr_all = gt_actions_arr
                    else:
                        gt_actions_arr_all = np.concatenate((gt_actions_arr_all, gt_actions_arr), axis=0)


                image = None

                ### prepare for next chunk action prediction
                self.val_dataloader.dataset.fix_sidx += self.args.data['train']['action_chunk']
                self.val_dataloader.dataset.fix_mem_idx = x = (np.linspace(0, self.val_dataloader.dataset.fix_sidx-1, self.args.data['train']['n_previous']).round().astype(np.int16)).tolist()


            occlusion_marks_sorted = []
            if self.args.return_action:
                occlusion_marks_sorted = sorted(occlusion_marks)
                x_axis = np.arange(gt_actions_arr_all.shape[0])
                num_dims = gt_actions_arr_all.shape[-1]
                for dim_idx in range(num_dims):
                    
                    ax = axes[dim_idx]

                    # Plot the continuous action sequences
                    ax.plot(x_axis, gt_actions_arr_all[:, dim_idx], label='Ground Truth', color='cornflowerblue', alpha=0.9)
                    ax.plot(x_axis, pd_actions_arr_all[:, dim_idx], label='Inferred', color='tomato', linestyle='--', alpha=0.9)

                    # Mark the starting point of each inference sequence
                    start_indices = np.arange(0, gt_actions_arr_all.shape[0], self.args.data["train"]["action_chunk"])

                    ax.scatter(start_indices, gt_actions_arr_all[start_indices, dim_idx], c='blue', marker='o', s=40, zorder=5, label='GT Start')

                    ax.scatter(start_indices, pd_actions_arr_all[start_indices, dim_idx], c='darkred', marker='x', s=40, zorder=5, label='Inferred Start')

                    for mark_idx, mark in enumerate(occlusion_marks_sorted):
                        label = None
                        if mark_idx == 0:
                            label = "Occlusion start"
                            if occlusion_label:
                                label += f" ({occlusion_label})"
                        ax.axvline(mark, color="#FFBF00", linestyle="--", linewidth=1.4, label=label)

                    # ax.set_title(f'Action Dimension {dim_idx}')
                    ax.set_title(f"Dimension- {dim_idx}")

                ax.set_ylabel('Value')
                ax.grid(True, linestyle=':', alpha=0.6)
                ax.legend()

                # Set common X-axis label
                fig.supxlabel(f'Continuous Timestep (across {n_chunk_action} inferences)')
                
                plt.tight_layout(rect=[0, 0, 1, 0.98]) # Adjust layout to make space for suptitle
                fig.suptitle(f'Comparison of Ground Truth and Inferred Actions', fontsize=18)
                
                plt.savefig(f'{self.save_folder}/openloop_evaluation_val{i_validation}.png', dpi=300, bbox_inches='tight')
                plt.clf()

            print(f"[DEBUG] Completed processing episode {i_validation}/{max_iterations-1}")
            print(f"[DEBUG] Occlusion markers (pred timesteps): {occlusion_marks_sorted}")
        
        print(f"[DEBUG] Finished processing all {max_iterations} episodes. Results saved to: {model_save_dir}")

    def infer(
        self,
        n_chunk_action=10,
        n_chunk_video=1,
        n_validation=10,
        global_step=0,
        domain_name="agibotworld",
        tasks_per_run=None,
        episodes_per_task=1,
        statistics_domain=None,
    ):
        print(
            f"[DEBUG infer] Received parameters: n_chunk_action={n_chunk_action}, "
            f"n_chunk_video={n_chunk_video}, n_validation={n_validation}, "
            f"domain_name={domain_name}, tasks_per_run={tasks_per_run}, "
            f"episodes_per_task={episodes_per_task}, statistics_domain={statistics_domain}"
        )
        model_save_dir = os.path.join(self.save_folder,f'Inference')
        self.validate(
            model_save_dir, global_step,
            n_view=len(self.args.data["train"]["valid_cam"]),
            n_chunk_video=n_chunk_video,
            n_chunk_action=n_chunk_action,
            n_validation=n_validation,
            domain_name=domain_name,
            tasks_per_run=tasks_per_run,
            episodes_per_task=episodes_per_task,
            statistics_domain=statistics_domain,
        )

    def rollout(
        self,
        rollout_steps=150,
        n_validation=1,
        domain_name="agibotworld",
        tasks_per_run=None,
        episodes_per_task=1,
        statistics_domain=None,
    ):
        if rollout_steps <= 0:
            raise ValueError("rollout_steps must be > 0 when calling rollout()")

        if not getattr(self, "val_dataloader", None):
            raise RuntimeError("Call prepare_val_dataset() before rollout.")
        if not getattr(self, "pipeline_class", None):
            raise RuntimeError("Call prepare_models() before rollout.")

        rollout_dir = os.path.join(self.save_folder, "Rollout")
        os.makedirs(rollout_dir, exist_ok=True)

        pipe = self.pipeline_class(
            self.scheduler, self.vae, self.text_encoder, self.tokenizer, self.diffusion_model
        )

        if not self.args.return_action:
            raise ValueError("return_action must be True for rollout.")

        action_type = self.args.data["train"]["action_type"]
        action_space = self.args.data["train"]["action_space"]
        chunk_size = self.args.data["train"]["action_chunk"]
        total_chunks = max(1, math.ceil(rollout_steps / chunk_size))
        n_prev = self.args.data["train"]["n_previous"]

        dataset = self.val_dataloader.dataset
        task_spans = getattr(dataset, "task_spans", [])
        total_tasks_available = len(task_spans)

        episodes_per_task = max(1, int(episodes_per_task))
        if total_tasks_available == 0:
            tasks_per_run_effective = 1
        else:
            if tasks_per_run is None or tasks_per_run <= 0:
                tasks_per_run_effective = math.ceil(n_validation / episodes_per_task)
            else:
                tasks_per_run_effective = tasks_per_run
            tasks_per_run_effective = max(1, min(tasks_per_run_effective, total_tasks_available))

        max_iterations = (
            min(n_validation, tasks_per_run_effective * episodes_per_task)
            if total_tasks_available
            else n_validation
        )

        print(
            f"[DEBUG rollout] rollout_steps={rollout_steps}, total_chunks={total_chunks}, "
            f"tasks_per_run_effective={tasks_per_run_effective}, max_iterations={max_iterations}"
        )
        print(f"[DEBUG rollout] Occlusion params: view={getattr(self.args, 'occlude_view', None)}, "
              f"start={getattr(self.args, 'occlude_start', None)}, end={getattr(self.args, 'occlude_end', None)}")

        for i_validation in range(max_iterations):
            if total_tasks_available:
                task_idx = i_validation // episodes_per_task
                episode_offset = i_validation % episodes_per_task
                span = task_spans[task_idx]
                span_end = span["start"] + span["length"] - 1
                target_idx = min(span["start"] + episode_offset, span_end)
                dataset.fix_epiidx = target_idx
                current_task = span["task"]
            else:
                dataset.fix_epiidx = i_validation
                current_task = None

            preds_all = []
            gts_all = []
            inference_points = []
            collected_steps = 0
            requested_step = 0
            occlusion_marks_all = []

            prev_start_index = None

            for chunk_idx in range(total_chunks):
                if collected_steps >= rollout_steps:
                    break

                remaining = rollout_steps - collected_steps
                step_data = dataset.get_step_data(
                    dataset.fix_epiidx,
                    requested_step,
                    rollout_horizon=chunk_size,
                )

                # If the dataset can no longer advance the starting index (e.g., because we are
                # too close to the end of the episode and indices are being clipped), then
                # continuing would just repeat (or even go backwards in) time. In that case,
                # we stop rollout early instead of automatically repeating the last segment.
                current_start_index = int(step_data.get("start_index", requested_step))
                if prev_start_index is not None and current_start_index <= prev_start_index:
                    print(
                        f"[DEBUG rollout] Detected non-increasing start_index "
                        f"(prev={prev_start_index}, current={current_start_index}); "
                        f"stopping rollout early to avoid repeating tail."
                    )
                    break
                prev_start_index = current_start_index

                video = step_data["video"].unsqueeze(0)
                actions_tensor = step_data["actions"].unsqueeze(0)
                caption = [step_data["caption"]]
                state_tensor = step_data["state"].unsqueeze(0)

                # Apply occlusion to the local clip if it intersects with history frames
                occlude_view = getattr(self.args, "occlude_view", None)
                occlude_start = getattr(self.args, "occlude_start", None)
                occlude_end = getattr(self.args, "occlude_end", None)
                n_prev_local = self.args.data['train']['n_previous']
                if occlude_view is not None and occlude_start is not None:
                    # Map occlusion window (global) -> local indices of this clip
                    start_idx_global = int(step_data["start_index"])
                    mem_start_global = start_idx_global - n_prev_local
                    # History window in global coordinates: [mem_start_global, start_idx_global)
                    hist_g_start = mem_start_global
                    hist_g_end = start_idx_global
                    # Compute intersection with [occlude_start, occlude_end)
                    occ_g_end = int(occlude_end) if occlude_end is not None else None
                    if occ_g_end is None:
                        inter_start = max(occlude_start, hist_g_start)
                        inter_end = hist_g_end
                    else:
                        inter_start = max(occlude_start, hist_g_start)
                        inter_end = min(occ_g_end, hist_g_end)
                    if inter_end > inter_start:
                        # Local indices within the n_previous window
                        rel_start = max(0, inter_start - hist_g_start)
                        rel_end = min(n_prev_local, inter_end - hist_g_start)
                        # Resolve view index
                        view_idx = None
                        if isinstance(occlude_view, str):
                            valid_cam = getattr(dataset, "valid_cam", None)
                            if valid_cam and occlude_view in valid_cam:
                                view_idx = valid_cam.index(occlude_view)
                            elif occlude_view.isdigit():
                                view_idx = int(occlude_view)
                        else:
                            try:
                                view_idx = int(occlude_view)
                            except (TypeError, ValueError):
                                view_idx = None
                        if view_idx is not None and 0 <= view_idx < video.shape[2]:
                            video[:, :, view_idx, rel_start:rel_end, ...] = 0.0
                            try:
                                print(
                                    f"[DEBUG rollout] Applied history occlusion on view={occlude_view} (idx={view_idx}); "
                                    f"global=[{occlude_start},{occlude_end}) ∩ history=[{hist_g_start},{hist_g_end}) "
                                    f"-> local=[{rel_start},{rel_end}) at step_start={start_idx_global}"
                                )
                            except Exception:
                                pass

                image = video[:, :, :, :n_prev_local]
                gt_video = video
                prompt = caption
                b, c, v, t, h, w = image.shape
                batch_size = 1
                image = rearrange(image, 'b c v t h w -> (b v) c t h w')

                if getattr(self.args, "add_state", False):
                    history_action_state = state_tensor[:batch_size]
                else:
                    history_action_state = None
                # Export the image to examine (with error handling for disk quota issues)
                try:
                    save_image(image, f'{rollout_dir}/image_{i_validation}_{chunk_idx}.png')
                    print(f"[DEBUG rollout] Saved image to {rollout_dir}/image_{i_validation}_{chunk_idx}.png")
                except OSError as e:
                    if e.errno == 122:  # Disk quota exceeded
                        print(f"[WARN rollout] Skipping image save due to disk quota exceeded")
                    else:
                        print(f"[WARN rollout] Failed to save image: {e}")
                except Exception as e:
                    print(f"[WARN rollout] Failed to save image: {e}")
                preds = pipe.infer(
                    image=image,
                    prompt=prompt[:batch_size],
                    negative_prompt='',
                    num_inference_steps=self.args.num_inference_step,
                    decode_timestep=0.03,
                    decode_noise_scale=0.025,
                    guidance_scale=1.0,
                    height=h,
                    width=w,
                    n_view=v,
                    return_action=True,
                    n_prev=self.args.data['train']['n_previous'],
                    chunk=(self.args.data['train']['chunk'] - 1) // self.TEMPORAL_DOWN_RATIO + 1,
                    return_video=False,
                    noise_seed=42,
                    action_chunk=chunk_size,
                    history_action_state=history_action_state,
                    pixel_wise_timestep=self.args.pixel_wise_timestep,
                    n_chunk=1,
                    action_dim=self.args.diffusion_model["config"]["action_in_channels"],
                )[0]

                fake_batch = {
                    "actions": actions_tensor,
                    "state": state_tensor,
                }
                pd_actions_arr, gt_actions_arr = self._denormalize_actions(
                    fake_batch, preds['action'][0], domain_name, action_type, action_space, statistics_domain
                )

                take = min(remaining, pd_actions_arr.shape[0])

                inference_points.append(collected_steps)
                preds_all.append(pd_actions_arr[:take])
                gts_all.append(gt_actions_arr[:take])

                # Record occlusion start marker along the continuous prediction timeline
                before_collected = collected_steps
                collected_steps += take
                requested_step = step_data["start_index"] + take
                if occlude_start is not None:
                    chunk_start_global = int(step_data["start_index"])
                    chunk_end_global = chunk_start_global + take
                    mark_local = None
                    if occlude_start < chunk_start_global and (occlude_end is None or occlude_end > chunk_start_global):
                        mark_local = 0
                    elif chunk_start_global <= occlude_start < chunk_end_global:
                        mark_local = occlude_start - chunk_start_global
                    if mark_local is not None:
                        occlusion_marks_all.append(before_collected + int(mark_local))

                if collected_steps >= rollout_steps:
                    break

            preds_all = np.concatenate(preds_all, axis=0) if preds_all else np.zeros((0, chunk_size))
            gts_all = np.concatenate(gts_all, axis=0) if gts_all else np.zeros((0, chunk_size))

            info = {
                "pred_action_across_time": preds_all,
                "gt_action_across_time": gts_all,
                "traj_id": dataset.fix_epiidx,
                "mse": float(np.mean((gts_all - preds_all) ** 2)) if preds_all.size else 0.0,
                "action_dim": preds_all.shape[1] if preds_all.size else 0,
                "action_horizon": chunk_size,
                "steps": preds_all.shape[0],
                "inference_points": inference_points,
                "modality_desc": action_space,
                "occlusion_marks": sorted(set(occlusion_marks_all)),
                "occlusion_label": getattr(self.args, "occlude_view", None),
            }

            filename_task = (
                f"task_{current_task}" if current_task else f"episode_{dataset.fix_epiidx}"
            )
            save_path = os.path.join(
                rollout_dir, f"rollout_{filename_task}_{i_validation}.png"
            )
            if info["action_dim"] > 0 and info["steps"] > 0:
                print(f"[DEBUG rollout] Occlusion markers: {info['occlusion_marks']}")
                self._plot_rollout(info, save_plot_path=save_path)
                np.savez_compressed(
                    os.path.join(rollout_dir, f"rollout_{filename_task}_{i_validation}.npz"),
                    pred=preds_all,
                    gt=gts_all,
                    inference_points=np.array(inference_points),
                )
            else:
                print(f"[WARN rollout] No data collected for rollout {i_validation}.")
