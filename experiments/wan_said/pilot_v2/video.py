import re

import av
import numpy as np
import torch
from torch.nn import functional as F

from ..io_utils import sha256
from .config import digest, signature


def frame_timeline(path):
    times = []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.codec_context.thread_count = 1
        fps = float(stream.average_rate or 0)
        for frame in container.decode(video=0):
            if frame.time is None:
                raise ValueError('Missing video frame PTS')
            times.append(float(frame.time))
        height, width = stream.height, stream.width
    if len(times) < 2 or fps <= 0 or np.any(np.diff(times) < 0) or times[-1] <= times[0]:
        raise ValueError('Invalid or nonmonotonic video timeline')
    return np.asarray(times), dict(source_fps=fps, source_frames=len(times),
                                  source_height=height, source_width=width)


def full_time_indices(times, count):
    if count < 2:
        raise ValueError('At least two samples are required')
    wanted = np.linspace(times[0], times[-1], count)
    right = np.searchsorted(times, wanted).clip(0, len(times)-1)
    left = np.maximum(right-1, 0)
    indices = np.where(np.abs(times[right]-wanted) < np.abs(times[left]-wanted), right, left)
    indices[0], indices[-1] = 0, len(times)-1
    return wanted, indices


def crop_resize(images, height, width):
    h, w = images.shape[-2:]
    if h * width > w * height:
        ch, cw = max(1, int(w * height / width)), w
    else:
        ch, cw = h, max(1, int(h * width / height))
    top, left = int(round((h-ch)/2)), int(round((w-cw)/2))
    cropped = images[..., top:top+ch, left:left+cw]
    resized = F.interpolate(cropped, size=(height, width), mode='bilinear',
                            align_corners=True, antialias=True).clamp(0, 1)
    return resized, dict(crop_box_xywh=[left, top, cw, ch],
                         retained_area_fraction=ch*cw/(h*w),
                         resize_axis_scale_ratio=(height/ch)/(width/cw),
                         crop_removes_content=(ch < h or cw < w))


def sample_video(row, cfg):
    if sha256(row['video']) != row['sha256']:
        raise ValueError('Source video SHA256 changed')
    times, metadata = frame_timeline(row['video'])
    wanted, indices = full_time_indices(times, cfg['frames'])
    requested = set(indices.tolist())
    selected = {}
    with av.open(row['video']) as container:
        container.streams.video[0].codec_context.thread_count = 1
        for i, frame in enumerate(container.decode(video=0)):
            if i in requested:
                selected[i] = frame.to_ndarray(format='rgb24')
    images = torch.from_numpy(np.stack([selected[i] for i in indices])).permute(0, 3, 1, 2).float()/255
    images, crop = crop_resize(images, cfg['height'], cfg['width'])
    actual = times[indices]
    span = float(times[-1]-times[0])
    metadata.update(crop, source_video_id=row['source_id'], source_video_sha256=row['sha256'],
                    source_caption_hash=digest(row['caption']), caption=row['caption'],
                    caption_scope='full_source_video', source_duration=span+1/metadata['source_fps'],
                    semantic_time_range=[float(times[0]), float(times[-1])],
                    wan_requested_timestamps=wanted.tolist(), wan_sampled_timestamps=actual.tolist(),
                    teacher_sampled_timestamps=actual[1:].tolist(),
                    wan_effective_fps=(len(actual)-1)/span,
                    teacher_sampling_policy='same_wan_sequence_remove_first_frame',
                    teacher_wan_frame_indices=list(range(1, len(actual))),
                    temporal_coverage=float((actual[-1]-actual[0])/span),
                    teacher_temporal_coverage=float((actual[-1]-actual[1])/span),
                    unique_sampled_frames=len(requested), max_sample_gap_seconds=float(np.diff(actual).max()),
                    sampling_may_miss_actions=True,
                    multi_stage_caption=bool(re.search(r'\b(then|before|after|finally|subsequently|while)\b', row['caption'], re.I)),
                    preprocessing_hash=signature(cfg))
    return images, metadata


def teacher_input(images, cfg):
    if images.shape != (cfg['frames'], 3, cfg['height'], cfg['width']):
        raise ValueError('Teacher requires the exact shared Wan crop and timeline')
    mean = images.new_tensor(cfg['normalization_mean'])[None, :, None, None]
    std = images.new_tensor(cfg['normalization_std'])[None, :, None, None]
    normalized = (images[1:]-mean)/std
    video = F.interpolate(normalized, (cfg['teacher_height'], cfg['teacher_width']),
                          mode='bicubic', align_corners=False)
    return video.permute(1, 0, 2, 3).unsqueeze(0).contiguous()
