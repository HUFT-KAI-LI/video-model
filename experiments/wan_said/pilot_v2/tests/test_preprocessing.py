import tempfile
import unittest
from pathlib import Path

import av
import numpy as np
import torch

from experiments.wan_said.io_utils import sha256
from experiments.wan_said.pilot_v2.config import load_config, signature, validate
from experiments.wan_said.pilot_v2.teacher import (feature_metadata, interpolate_positions,
                                                  validate_feature_metadata)
from experiments.wan_said.pilot_v2.video import crop_resize, full_time_indices, sample_video, teacher_input


class PreprocessingTests(unittest.TestCase):
    def test_irregular_pts_endpoint_coverage_and_duplicates_explicit(self):
        times = np.array([1.2, 1.3, 1.8, 3.0])
        wanted, index = full_time_indices(times, 49)
        self.assertEqual((index[0], index[-1]), (0, 3))
        self.assertEqual((wanted[0], wanted[-1]), (1.2, 3.0))
        self.assertTrue(np.all(np.diff(index) >= 0))

    def test_shared_crop_not_stretch_and_teacher_uses_same_pixels(self):
        cfg = load_config()
        image = torch.rand(49, 3, 64, 128)
        cropped, report = crop_resize(image, cfg['height'], cfg['width'])
        self.assertEqual(report['crop_box_xywh'], [16, 0, 96, 64])
        self.assertAlmostEqual(report['resize_axis_scale_ratio'], 1)
        video = teacher_input(cropped, cfg)
        self.assertEqual(tuple(video.shape), (1, 3, 48, 160, 240))
        mean = torch.tensor(cfg['normalization_mean'])[None, :, None, None]
        std = torch.tensor(cfg['normalization_std'])[None, :, None, None]
        expected = torch.nn.functional.interpolate((cropped[1:]-mean)/std, (160, 240),
                                                   mode='bicubic', align_corners=False)
        self.assertTrue(torch.equal(video[0].permute(1, 0, 2, 3), expected))

    def test_position_identity_and_parameterized_rectangular_time(self):
        positions = torch.rand(1, 8*14*14, 12)
        self.assertTrue(torch.equal(interpolate_positions(positions, (8, 14, 14), (8, 14, 14)), positions))
        for time in (8, 24, 40):
            out = interpolate_positions(positions, (8, 14, 14), (time, 10, 15))
            self.assertEqual(tuple(out.shape), (1, time*150, 12))
            self.assertTrue(torch.isfinite(out).all())
        with self.assertRaises(ValueError):
            interpolate_positions(positions[:, :-1], (8, 14, 14), (24, 10, 15))

    def test_invalid_wan_frames_aspect_small_teacher_and_cache_rejected(self):
        cfg = load_config()
        for update in ({'frames': 48}, {'teacher_width': 224}, {'teacher_dim': 384},
                       {'feature_cache_enabled': True}, {'width': 719}):
            with self.assertRaises(ValueError):
                validate(dict(cfg, **update))

    def test_feature_identity_roundtrip_and_old_small_cache_rejected(self):
        cfg = load_config()
        row = dict(sha256='source-sha', caption='A complete action.')
        metadata = feature_metadata(cfg, row)
        validate_feature_metadata(metadata, cfg, row)
        for update in ({'teacher_hidden_dim': 384}, {'source_video_sha256': 'other'},
                       {'source_caption_hash': 'other'}, {'teacher_input_frames': 16}):
            with self.assertRaises(ValueError):
                validate_feature_metadata(dict(metadata, **update), cfg, row)
        changed = dict(cfg, frames=81)
        self.assertNotEqual(signature(cfg), signature(changed))

    def test_real_fixture_full_clip_ignores_old_middle_window(self):
        cfg = dict(load_config(), frames=17, height=32, width=48)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'video.mp4'
            with av.open(str(path), 'w') as output:
                stream = output.add_stream('libx264', rate=8)
                stream.width, stream.height, stream.pix_fmt = 96, 64, 'yuv420p'
                for i in range(24):
                    frame = av.VideoFrame.from_ndarray(np.full((64, 96, 3), i*8, np.uint8), format='rgb24')
                    for packet in stream.encode(frame):
                        output.mux(packet)
                for packet in stream.encode():
                    output.mux(packet)
            row = dict(video=str(path), sha256=sha256(path), source_id='fixture',
                       caption='First it darkens, then it becomes bright.', window_start=1.0)
            images, sampling = sample_video(row, cfg)
            self.assertEqual(images.shape, (17, 3, 32, 48))
            self.assertEqual(sampling['wan_sampled_timestamps'][0], 0)
            self.assertEqual(sampling['wan_sampled_timestamps'][-1], 23/8)
            self.assertEqual(sampling['temporal_coverage'], 1)
            self.assertEqual(sampling['teacher_sampled_timestamps'], sampling['wan_sampled_timestamps'][1:])
            self.assertTrue(sampling['multi_stage_caption'])


if __name__ == '__main__':
    unittest.main()
