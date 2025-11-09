
import sys
import os
import io
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import traceback
import json
import random
import math
import numpy as np
import pandas as pd

import torch
from torch.utils.data.dataset import Dataset
from einops import rearrange
import glob
from moviepy.editor import VideoFileClip
import torchvision.transforms as transforms
from tqdm import tqdm
import torch.nn.functional as F
import cv2
from PIL import Image

from data.utils.domain_table import DomainTable
from data.utils.statistics import StatisticInfo
# from data.utils.get_actions import parse_h5

from utils import zero_rank_print
from data.utils.utils import intrinsic_transform, gen_crop_config, intrin_crop_transform

def load_jsonl(jsonl_path):
    """
    load jsonl file
    """
    data = []
    if os.path.exists(jsonl_path):
        with open(jsonl_path, 'r', encoding='UTF-8') as f:
            for line in f:
                data.append(json.loads(line))
    return data


class CustomLeRobotDataset(Dataset):
    def __init__(self,
        data_roots,
        domains,
        task_recap_file = None,
        step_recap_file = None,
        sample_size=(192, 256), 
        sample_n_frames=64,
        preprocess = 'resize',
        valid_cam = ['observation.images.top_head', 'observation.images.hand_left', 'observation.images.hand_right'],
        chunk=1,
        action_chunk=None,
        n_previous=-1,
        previous_pick_mode='uniform',
        random_crop=True,
        dataset_info_cache_path = None,
        action_type = "absolute",
        action_space = "joint",
        train_dataset=True,
        action_key = "action",
        state_key = "observation.state",
        use_unified_prompt = False,
        unified_prompt = "best quality, consistent and smooth motion, realistic, clear and distinct.",
        fix_epiidx = None,
        fix_sidx = None,
        fix_mem_idx = None,
        stat_file = None,
    ):
        """
        data_roots:              directory of LeRoBot dataset
        domains:                 name of your dataset, used to index different statistics
        task_recap_file:         json file of augmented task captions:
                                 {
                                    'ori_task_caption_1': ['new_caption_1', 'new_caption_2'...],
                                    'ori_task_caption_2': ['new_caption_1', 'new_caption_2'...],
                                 }
        step_recap_file:         json file of augmented step captions:
                                 {
                                    'ori_step_caption_1': ['new_caption_1', 'new_caption_2'...],
                                    'ori_step_caption_2': ['new_caption_1', 'new_caption_2'...],
                                 }
        sample_size:             video frame size
        sample_n_frames:         number of frames used to randomly or uniformly select memories
        preprocess:              frame preprocessing strategy, resize or center_crop_resize
        valid_cam:               list of cam names 
        chunk:                   number of video frames to predict
        action_chunk:            number of actions to predict, action_chunk should be an integer multiple of chunk.
        n_previous:              number of memory frames
        previous_pick_mode:      how to select memories
        random_crop:             randomly crop images
        dataset_info_cache_path: path to save dataset meta information cache
        action_type:             action space to use in this dataset
                                    'absolute': norm(act_t)
                                    'delta':    norm(act_t - act_{t-1})
                                    'relative': norm(act_t) - norm(state)
        action_space:            joint or eef, which is used to determinate the statistics values only in this dataset
        ignore_seek:             if True, load the first furture frame only
        use_unified_prompt:      if set all prompt the same
        unified_prompt:          unified prompt
        fix_epiidx:              used in validation stage only, set episode index to fix_epiidx
        fix_sidx:                used in validation stage only, set start index to fix_sidx
        fix_mem_idx:             used in validation stage only, set memory indexes to fix_mem_idx
        """
        
        zero_rank_print(f"loading annotations...")

        def append_dataset_from_meta(root_prefix, domain_dir, domain_key):
            meta_folder = os.path.join(root_prefix, domain_dir, "meta")
            data_folder = os.path.join(root_prefix, domain_dir, "data")
            video_folder = os.path.join(root_prefix, domain_dir, "videos")
            if not os.path.exists(data_folder):
                zero_rank_print(f"data folder not found: {data_folder}")
                return

            tasks_jsonl = os.path.join(meta_folder, "tasks.jsonl")
            task_entries = load_jsonl(tasks_jsonl)
            default_task_name = domain_dir
            if not task_entries:
                task_entries = [{"task_index": 0, "task": default_task_name}]
            task_index_task_str_dict = {}
            for item in task_entries:
                task_idx = item.get("task_index", len(task_index_task_str_dict))
                task_name = item.get("task") or item.get("task_name") or item.get("task_str") or default_task_name
                task_index_task_str_dict[task_idx] = task_name

            info_path = os.path.join(meta_folder, "info.json")
            total_chunks = None
            chunks_size = None
            if os.path.exists(info_path):
                with open(info_path, "r") as f:
                    metainfo = json.load(f)
                    total_chunks = metainfo.get("total_chunks")
                    chunks_size = metainfo.get("chunks_size")

            episodes_jsonl = os.path.join(meta_folder, "episodes.jsonl")
            epiosdes_data = load_jsonl(episodes_jsonl)
            if not epiosdes_data:
                epiosdes_data = []
                chunk_dirs = sorted(glob.glob(os.path.join(data_folder, "chunk-*")))
                for chunk_dir in chunk_dirs:
                    chunk_idx = int(os.path.basename(chunk_dir).split('-')[-1])
                    parquet_files = sorted(glob.glob(os.path.join(chunk_dir, "episode_*.parquet")))
                    for pq in parquet_files:
                        episode_index = int(os.path.splitext(os.path.basename(pq))[0].split('_')[-1])
                        epiosdes_data.append({
                            "episode_index": episode_index,
                            "tasks": [task_index_task_str_dict.get(0, domain_key)],
                            "length": 0,
                            "chunk": chunk_idx,
                        })

            chunk_dirs_cache = sorted(glob.glob(os.path.join(data_folder, "chunk-*")))

            for episode_data in tqdm(epiosdes_data):
                episode_index = episode_data.get('episode_index')
                if episode_index is None:
                    continue
                tasks = episode_data.get('tasks') or [task_index_task_str_dict.get(0, domain_key)]
                if len(tasks) > 1:
                    task_idx = random.choice(tasks)
                else:
                    task_idx = tasks[0]
                task = task_index_task_str_dict.get(task_idx, default_task_name)
                length = episode_data.get('length', sample_n_frames)
                if not length:
                    length = sample_n_frames

                candidate_chunks = []
                if episode_data.get("chunk") is not None:
                    candidate_chunks.append(int(episode_data["chunk"]))
                if chunks_size:
                    candidate_chunks.append(int(episode_index // max(chunks_size, 1)))
                if not candidate_chunks:
                    candidate_chunks.extend([int(os.path.basename(cd).split('-')[-1]) for cd in chunk_dirs_cache])
                candidate_chunks = list(dict.fromkeys(candidate_chunks))

                parquet_path = None
                episode_chunk = None
                for chunk_idx in candidate_chunks:
                    candidate_path = os.path.join(data_folder, f"chunk-{int(chunk_idx):03d}", f"episode_{episode_index:06d}.parquet")
                    if os.path.exists(candidate_path):
                        parquet_path = candidate_path
                        episode_chunk = int(chunk_idx)
                        break
                if parquet_path is None:
                    for cd in chunk_dirs_cache:
                        candidate_path = os.path.join(cd, f"episode_{episode_index:06d}.parquet")
                        if os.path.exists(candidate_path):
                            parquet_path = candidate_path
                            episode_chunk = int(os.path.basename(cd).split('-')[-1])
                            break
                if parquet_path is None:
                    zero_rank_print(f"parquet file not found: {episode_index}")
                    continue

                video_path = os.path.join(video_folder, f"chunk-{episode_chunk:03d}", "{}", f"episode_{episode_index:06d}.mp4")
                domain_id = DomainTable.get(domain_key, -1)

                info = [
                    video_path,
                    None,
                    parquet_path,
                    domain_key, domain_id,
                    None, task,
                    length,
                ]

                self.dataset.append(info)

        assert(action_type in ["delta", "absolute", "relative"])
        self.action_type = action_type
        assert(action_space in ["eef", "joint"])
        self.action_space = action_space



        self.action_key = action_key
        self.state_key = state_key

        self.random_crop = random_crop
        
        if not isinstance(valid_cam, (list, tuple)):
            valid_cam = [valid_cam, ]
        self.valid_cam = valid_cam
        if len(data_roots) == 1 and len(domains) > 1:
            data_roots = data_roots * len(domains)
        self.data_roots = data_roots
        self.dataset = []
        
        if dataset_info_cache_path is not None and os.path.exists(dataset_info_cache_path):
            zero_rank_print(f"Load Cache Dataset Information from {dataset_info_cache_path}")
            with open(dataset_info_cache_path, "r") as f:
                self.dataset = json.load(f)
        else:
            # construct the dataset_info
            for _data_root, _domain_name in zip(self.data_roots, domains):

                print(f"Loading {_domain_name} data from {_data_root}")
                
                meta_folder = os.path.join(_data_root, _domain_name, "meta")
                if os.path.exists(meta_folder):
                    append_dataset_from_meta(_data_root, _domain_name, _domain_name)
                else:
                    domain_root = os.path.join(_data_root, _domain_name)
                    sub_domains = []
                    if os.path.isdir(domain_root):
                        sub_domains = sorted([d for d in os.listdir(domain_root)
                                              if os.path.isdir(os.path.join(domain_root, d))])
                    if not sub_domains:
                        zero_rank_print(f"No sub-domains found in {domain_root}")
                        continue
                    for sub_domain in sub_domains:
                        append_dataset_from_meta(domain_root, sub_domain, _domain_name)

        if dataset_info_cache_path is not None and not(os.path.exists(dataset_info_cache_path)):
            zero_rank_print(f"Save Cache Dataset Information to {dataset_info_cache_path}")
            cache_dir = os.path.dirname(dataset_info_cache_path)
            if cache_dir:
                os.makedirs(cache_dir, exist_ok=True)
            with open(dataset_info_cache_path, "w") as f:
                json.dump(self.dataset, f)

        self.length = len(self.dataset)
        zero_rank_print(f"data scale: {self.length}")

        # Pre-compute contiguous episode spans for each task caption
        self.task_spans = []
        self.task_to_span = {}
        if self.length > 0:
            prev_task = None
            span_start = 0
            for idx, info in enumerate(self.dataset):
                task_name = str(info[6])
                if task_name != prev_task:
                    if prev_task is not None:
                        span = {
                            "task": prev_task,
                            "start": span_start,
                            "length": idx - span_start,
                        }
                        self.task_spans.append(span)
                        self.task_to_span[prev_task] = span
                    prev_task = task_name
                    span_start = idx
            if prev_task is not None and span_start < self.length:
                span = {
                    "task": prev_task,
                    "start": span_start,
                    "length": self.length - span_start,
                }
                self.task_spans.append(span)
                self.task_to_span[prev_task] = span
        self.total_tasks = len(self.task_spans)

        self.chunk = chunk
        if action_chunk is None:
            action_chunk = chunk
        self.action_chunk = action_chunk
        self.video_temporal_stride = self.action_chunk // self.chunk
        assert(self.chunk * self.video_temporal_stride == self.action_chunk)

        self.sample_n_frames = sample_n_frames
        
        self.sample_size = sample_size

        if preprocess == 'center_crop_resize':
            self.pixel_transforms_resize = transforms.Compose([
                transforms.Resize(min(sample_size)),  # the size of shape (1,) means the smaller edge will be resized to it and the img will keep the h-w ratio.
                transforms.CenterCrop(sample_size),
            ])
        if preprocess == 'resize':
            self.pixel_transforms_resize = transforms.Compose([
                transforms.Resize(sample_size),
            ])
        self.pixel_transforms_norm = transforms.Compose([
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ])
        self.preprocess = preprocess

        if n_previous > 1:
            self.n_previous = n_previous
            self.previous_pick_mode = previous_pick_mode
        else:
            self.n_previous = self.sample_n_frames - self.chunk
            self.previous_pick_mode = 'uniform'

        if task_recap_file is not None:
            with open(task_recap_file, 'r', encoding='UTF-8') as f:
                self.task_recap_map = json.load(f)
        else:
            self.task_recap_map = None

        if step_recap_file is not None:
            with open(step_recap_file, 'r', encoding='UTF-8') as f:
                self.step_recap_map = json.load(f)
        else:
            self.step_recap_map = None

        self.use_unified_prompt = use_unified_prompt

        ### validation only
        self.fix_epiidx = fix_epiidx
        self.fix_sidx = fix_sidx
        self.fix_mem_idx = fix_mem_idx

        ### load stat_file if provided
        self.StatisticInfo = StatisticInfo
        if stat_file is not None:
            with open(stat_file, "r") as f:
                self.StatisticInfo = json.load(f)

    def get_frame_indexes(self, total_frames, ):
        """
        select self.n_previous memory frames and self.action_chunk prediction frmaes
        1. randomly select the end frame
        2. take frames from {end-action_chunk} to {end} as the prediction frames
        3. uniformly/randomly select memory frames from {end-self.sample_n_frames} to {end-action_chunk}
        """

        if self.fix_sidx is not None and self.fix_mem_idx is not None:
            action_indexes = list(range(self.fix_sidx, self.fix_sidx+self.action_chunk))
            frame_indexes = action_indexes[::self.video_temporal_stride]
            return self.fix_mem_idx + frame_indexes, self.fix_mem_idx + action_indexes

        chunk_end = random.randint(self.action_chunk, total_frames+self.action_chunk)
        indexes = np.array(list(range(chunk_end-self.sample_n_frames, chunk_end)))
        indexes = np.clip(indexes, a_min=1, a_max=total_frames-1).tolist()
        video_end = indexes[-self.action_chunk:]
        mem_candidates = [
            indexes[int(i)] for i in range(0, self.sample_n_frames-self.action_chunk-1)
        ]

        if self.previous_pick_mode == 'uniform':
            mem_indexes = [mem_candidates[int(i)] for i in np.linspace(0, len(mem_candidates)-1, self.n_previous).tolist()]

        elif self.previous_pick_mode == 'random':
            mem_indexes = [mem_candidates[i] for i in sorted(np.random.choice(list(range(0,len(mem_candidates)-1)), size=self.n_previous-1, replace=False).tolist())] + [mem_candidates[-1]]

        else:
            raise NotImplementedError(f"unsupported previous_pick_mode: {self.previous_pick_mode}")       

        frame_indexes = mem_indexes + video_end[self.video_temporal_stride-1::self.video_temporal_stride]
        action_indexes = mem_indexes + video_end

        return frame_indexes, action_indexes


    def get_action_bias_std(self, domain_name):
        return torch.tensor(self.StatisticInfo[domain_name+"_"+self.action_space]['mean']).unsqueeze(0), torch.tensor(self.StatisticInfo[domain_name+"_"+self.action_space]['std']).unsqueeze(0)+1e-6


    def seek_mp4(self, video_path, cam_name_list, slices):
        """
        seek video frames according to the input slices;
        output video shape: (c,v,t,h,w)
        """
        video_list = []
        for cam_name in cam_name_list:
            video_reader = VideoFileClip(video_path.format(cam_name))
            fps = video_reader.fps
            video = []
            for idx in slices:
                video.append(video_reader.get_frame(float(idx)/fps))
            video = torch.from_numpy(np.stack(video)).permute(3, 0, 1, 2).contiguous()
            video = video.float()/255.
            if hasattr(self, "sample_size") and self.sample_size is not None:
                target_h, target_w = self.sample_size
                if video.shape[-2:] != (target_h, target_w):
                    frames = video.permute(1, 0, 2, 3)  # (t, c, h, w)
                    frames = F.interpolate(
                        frames,
                        size=(target_h, target_w),
                        mode="bilinear",
                        align_corners=False,
                    )
                    video = frames.permute(1, 0, 2, 3)
            video_reader.close()
            video_list.append(video)
        video_list = torch.stack(video_list, dim=1)
        return video_list



    def transform_video(self, videos, specific_transforms_resize, intrinsics, sample_size):
        """
        crop (optional) and resize the videos, and modify the intrinsic accordingly
        """
        c, v, t, h, w = videos.shape
        new_videos = []
        new_intrinsics = []
        for iv in range(v):
            video = videos[:, iv]
            if self.random_crop:
                h_start, w_start, h_crop, w_crop = gen_crop_config(video)
                video = video[:,:,h_start:h_start+h_crop,w_start:w_start+w_crop]
                if intrinsics is not None:
                    intrinsic = intrin_crop_transform(intrinsics[iv], h_start, w_start)
                
                h, w = h_crop, w_crop
            if intrinsics is not None:
                intrinsic = intrinsic_transform(intrinsic, (h, w), sample_size, self.preprocess)
                new_intrinsics.append(intrinsic)
                
            video = specific_transforms_resize(video)
            new_videos.append(video)
        new_videos = torch.stack(new_videos, dim=1)
        if len(new_intrinsics) > 0:
            new_intrinsics = torch.stack(new_intrinsics, dim=0)
        else:
            new_intrinsics = None
        return new_videos, None


    def normalize_video(self, video, specific_transforms_norm):
        """
        input video should have shape (c,v,t,h,w)
        """
        c,v,t,h,w = video.shape
        video = specific_transforms_norm(video.permute(1,2,0,3,4).reshape(-1,c,h,w)).reshape(v,t,c,h,w).permute(2,0,1,3,4)
        return video


    def get_transform(self, ):
        sample_size = self.sample_size
        specific_transforms_resize = self.pixel_transforms_resize
        specific_transforms_norm = self.pixel_transforms_norm
        return sample_size, specific_transforms_resize, specific_transforms_norm


    def get_long_recaption(self, step_captions, task_caption):
        newcap = []
        # find = []
        for step_caption in step_captions:
            if self.step_recap_map is not None:
                recap_list = self.step_recap_map.get(step_caption,[])
                recap_list.append(step_caption)
                step_caption = np.random.choice(recap_list,1)
                newcap.append(str(step_caption[0]))
            else:
                newcap.append(step_caption)

        newcap = ", ".join(newcap)
        newcap = newcap.replace(" the "," ")
        if self.task_recap_map is not None:
            task_recap_list = self.task_recap_map.get(task_caption,[])
            task_recap_list.append(task_caption)
            task_newcap = np.random.choice(task_recap_list,1)
            task_newcap = str(task_newcap[0])
            fullcap = task_newcap + ": " + newcap
        else:
            task_newcap = task_caption
            fullcap = task_caption + ": " + newcap
        cap_type = random.randint(0,2)
        allcap = [fullcap, task_newcap, newcap]
        recap = allcap[cap_type]
        return recap



    def get_batch(self, idx):
        
        video_path = self.dataset[idx][0]
        parquet_path = self.dataset[idx][2]
        domain_name = self.dataset[idx][3]
        domain_id = self.dataset[idx][4]
        caption = self.dataset[idx][6]
        # Ensure caption is always a string (handle cache loading issues)
        if not isinstance(caption, str):
            caption = str(caption) if caption is not None else ""
        total_frames = self.dataset[idx][7]
        
        sample_size, specific_transforms_resize, specific_transforms_norm = self.get_transform()
        vid_indexes, indexes = self.get_frame_indexes(total_frames, )
        
        data = pd.read_parquet(parquet_path)

        def pad_to_16(vec: torch.Tensor) -> torch.Tensor:
            """
            If channel dim is 14 (joints only), insert two gripper slots to align to 16 dims.
            Layout: 0..6 joints, 7 gripper_a (0), 8..14 joints, 15 gripper_b (0).
            vec: (..., C)
            """
            if vec.shape[-1] == 16:
                return vec
            if vec.shape[-1] == 14:
                left = vec[..., :7]
                right = vec[..., 7:]
                z = torch.zeros_like(left[..., :1])
                return torch.cat([left, z, right, z], dim=-1)
            return vec


        action_mean, action_std = self.get_action_bias_std(domain_name)
        state_mean, state_std = self.get_action_bias_std(domain_name + "_state")
        
        ###
        ### example data
        ### data[self.action_key] with the shape of T*C: [[1.0, 1.0, 1.0, ...], ...]
        ### data[self.state_key]  with the shape of T*C: [[1.0, 1.0, 1.0, ...], ...]
        action_len = data[self.action_key].shape[0]
        state_len = data[self.state_key].shape[0]
        try:
            action = np.stack([data[self.action_key][i] for i in range(action_len)])
            state = np.stack([data[self.state_key][i] for i in range(state_len)])
        except:
            raise ValueError("We currently only support action and state data with the shape of T*C!")

        if action_len > 0:
            max_action_idx = action_len - 1
            indexes = [max(0, min(max_action_idx, int(i))) for i in indexes]
        if total_frames > 0:
            max_video_idx = total_frames - 1
            vid_indexes = [max(0, min(max_video_idx, int(i))) for i in vid_indexes]

        state = torch.FloatTensor(state)[indexes][self.n_previous-1:self.n_previous]
        state = pad_to_16(state)
        state = (state - state_mean) / state_std

        if self.action_type == "absolute":
            ### act = norm(act)

            action = action[indexes].astype(np.float32)
            action = torch.FloatTensor(action)
            action = pad_to_16(action)
            action = (action - action_mean) / action_std

        elif self.action_type == "delta":
            ### delta_act = norm(act_{t} - act_{t-1})

            delta_act_meanv, delta_act_stdv = self.get_action_bias_std(domain_name + "_delta")
            action_curr = torch.FloatTensor(action[indexes].astype(np.float32))
            action_last = torch.FloatTensor(action[[_-1 for _ in indexes]].astype(np.float32))
            action_curr = pad_to_16(action_curr)
            action_last = pad_to_16(action_last)
            delta_action = action_curr - action_last
            ### keep current effector action
            delta_action[:, 6] = action_last[:, 6]
            delta_action[:, 13] = action_last[:, 13]
            delta_action = (delta_action - delta_act_meanv) / delta_act_stdv
            action = delta_action

        elif self.action_type == "relative":
            ### relative_act = norm(act) - norm(state)

            action_curr = action[indexes].astype(np.float32)
            action = torch.FloatTensor(action_curr)
            action = pad_to_16(action)
            action = (action - action_mean) / action_std
            rel_action = action - state
            action = rel_action

        else:

            raise NotImplementedError


        videos = self.seek_mp4(video_path, self.valid_cam, vid_indexes)

        videos, _ = self.transform_video(
            videos, specific_transforms_resize, None, sample_size
        )
        videos = self.normalize_video(videos, specific_transforms_norm)

        return videos, action, caption, state


    def __len__(self):
        return self.length


    def __getitem__(self, idx):        
        
        # video, actions, caption, state = self.get_batch(idx)

        if self.fix_epiidx is not None:
            video, actions, caption, state = self.get_batch(self.fix_epiidx)
        else:
            while True:
                try:
                    video, actions, caption, state = self.get_batch(idx)
                    break
                except:
                    ### print error information to debug
                    traceback.print_exc()
                    ### 
                    idx = random.randint(0, self.length-1)
                    
        sample = dict(
            video=video,
            actions=actions,
            caption=caption,
            state=state,
        )
        return sample
