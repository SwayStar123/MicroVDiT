# CelebVHQ metadata JSON format
# {
# "meta_info": 
#     {
#         "appearance_mapping": ["Blurry", "Male", "Young", ...],  // appearance attributes
#         "action_mapping": ["blow", "chew", "close_eyes", ...]    // action attributes
#     },  

# "clips": 
# {
#     "M2Ohb0FAaJU_1":  // clip 1 
#     {
#         "ytb_id": "M2Ohb0FAaJU",                                   // youtube id
#         "duration": {"start_sec": 81.62, "end_sec": 86.17},        // start and end times in the original video
#         "bbox": {"top": 0.0, "bottom": 0.8815, "left": 0.1964, "right": 0.6922},  // bounding box
#         "attributes":                                              // attributes information 
#         {
#             "appearance": [0, 0, 1, ...],                          // same order as the "appearance_mapping"
#             "action": [0, 0, 0, ...],                              // same order as the "action_mapping"
#             "emotion": {"sep_flag": false, "labels": "neutral"}    // only one emotion in the clip 
#          }, 
#          "version": "v0.1"
           
#     },
#     "_0tf2n3rlJU_0":  // clip 2 
#     {
#         "ytb_id": "_0tf2n3rlJU", 
#         "duration": {"start_sec": 52.72, "end_sec": 56.1}, 
#         "bbox": {"top": 0.0, "bottom": 0.8407, "left": 0.5271, "right": 1.0}, 
#         "attributes":                                              // attributes information (TBD)
#         {
#             "appearance": [0, 0, 1, ...], 
#             "action": [0, 0, 0, ...], 
#             "emotion": 
#             {
#                 "sep_flag": true, "labels": [                      // multi-emotion in the clip
#                     {"emotion": "neutral", "start_sec": 0, "end_sec": 0.28}, 
#                     {"emotion": "happy", "start_sec": 1.28, "end_sec": 3.28}]
#             }
#         }, 
#         "version": "v0.1" 
#     }
#     "..."
#     "..."

# }

import json
import os
import cv2
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from diffusers import AutoencoderKL

json_relative_path = "../../datasets/CelebV-HQ/celebvhq_info.json"
videos_relative_path = "../../datasets/CelebV-HQ/videos"

# Place holders
video_latent_mean = torch.tensor([0.0, 0.0, 0.0, 0.0])
video_latent_std = torch.tensor([1.0, 1.0, 1.0, 1.0])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

normalize_to_unit = transforms.Compose([
    transforms.ToTensor(),           # Convert to tensor
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # Scale to [-1, 1]
])

vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", cache_dir="../../models/vae")
vae = vae.to(device)

normalize = transforms.Compose([
    transforms.Normalize(video_latent_mean, video_latent_std)
])

# Denormalize back to original space
def denormalize(tensor):
    """
    Denormalizes a tensor using the provided mean and standard deviation.

    Args:
        tensor (torch.Tensor): The normalized tensor to denormalize.

    Returns:
        torch.Tensor: The denormalized tensor.
    """
    return tensor * video_latent_std.view(-1, 1, 1) + video_latent_mean.view(-1, 1, 1)

def default_transform(x):
    x = normalize_to_unit(x)
    x = vae.encode(x.to(device)).latent_dist.sample().cpu()
    x = normalize(x)
    return x

class CelebVHQDataset(Dataset):
    def __init__(self, json_path, videos_path, resize=None, target_fps=None, transform=None):
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        self.videos_path = videos_path
        self.resize = resize
        self.target_fps = target_fps
        self.transform = transform
        self.clip_ids = list(self.data['clips'].keys())

    def __len__(self):
        return len(self.clip_ids)

    def __getitem__(self, idx):
        clip_id = self.clip_ids[idx]
        video_file = os.path.join(self.videos_path, f"{clip_id}.mp4")

        # Read video using OpenCV
        cap = cv2.VideoCapture(video_file)
        frames = []
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = 1

        if self.target_fps is not None and original_fps > 0:
            frame_interval = int(round(original_fps / self.target_fps))
            if frame_interval < 1:
                frame_interval = 1

        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % frame_interval == 0:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if self.resize is not None:
                    frame = cv2.resize(frame, self.resize)
                if self.transform:
                    frame = self.transform(frame)
                frames.append(frame)
            frame_count += 1
        cap.release()

        if not frames:
            raise ValueError(f"No frames extracted from video {video_file}")
        
        # (D, C, H, W) -> (C, D, H, W)
        frames = torch.stack(frames).squeeze().transpose(0, 1)
        return frames

def collate_fn(batch, target_fps=None):
    if target_fps is not None:
        MAX_FRAMES = target_fps * 3
    else:
        MAX_FRAMES = 24 * 3

    # batch shape: (bs, C, D, H, W)
    # Find the minimum number of frames in the batch
    min_frames = min([x.size(1) for x in batch])

    if min_frames > MAX_FRAMES:
        min_frames = MAX_FRAMES

    # Trim all videos to the min_frames
    trimmed_batch = [x[:, :min_frames, :, :] for x in batch]
    # Stack into a tensor
    return torch.stack(trimmed_batch)

# Example usage:
# dataloader = get_celebvhq_dataloader(json_relative_path, videos_relative_path, batch_size=8, resize=(224, 224), target_fps=24)
# for batch in dataloader:
#     # batch shape: [batch_size, C, D, H, W]
#     pass
def get_celebvhq_dataloader(batch_size=4, shuffle=True, resize=None, target_fps=None, transform=default_transform, num_workers=4, json_path=json_relative_path, videos_path=videos_relative_path):
    dataset = CelebVHQDataset(json_path, videos_path, resize, target_fps, transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, 
                            collate_fn=lambda batch: collate_fn(batch, target_fps), num_workers=num_workers)
    return dataloader





