import json
import os
from glob import glob

import cv2 as cv
import numpy as np
import torch


_META_ALIASES = {
    'radar_los': ('radar_los',),
    'range_resolution': ('range_resolution',),
    'azimuth_resolution': ('azimuth_resolution',),
    'range_axis': ('range_axis',),
    'azimuth_axis': ('azimuth_axis',),
    'image_size': ('image_size',),  # range, azimuth
}

def _first_present(mapping, aliases, default=None):
    for key in aliases:
        if key in mapping:
            return mapping[key]
    return default


def _as_numpy(value, dtype=np.float32):
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        return value.astype(dtype) if dtype is not None else value
    return np.asarray(value, dtype=dtype)


def _normalize_vector(value, name):
    vec = _as_numpy(value, dtype=np.float32).reshape(-1)
    if vec.shape[0] != 3:
        raise ValueError('{} must be a 3-vector, got shape {}'.format(name, vec.shape))
    norm = np.linalg.norm(vec)
    if norm < 1e-8:
        raise ValueError('{} must be non-zero'.format(name))
    return vec / norm


def _select_frame_value(values, frame_idx, n_frames):
    arr = np.asarray(values)
    if arr.ndim > 0 and arr.shape[0] == n_frames:
        return arr[frame_idx]
    return arr



def _conf_get(conf, name, default=None):
    try:
        value = conf.get(name)
    except Exception:
        return default
    return default if value is None else value


def _conf_get_string(conf, name, default):
    try:
        return conf.get_string(name, default=default)
    except Exception:
        return default


def _conf_get_float(conf, name, default):
    try:
        return conf.get_float(name, default=default)
    except Exception:
        return default


class ISARDataset:
    """
    Dataset for ISAR image sequences.
    Strictly follows 1D axis definitions:
    Range = Vertical/Y, Azimuth = Horizontal/X.
    """

    def __init__(self, conf):
        super(ISARDataset, self).__init__()
        print('Load ISAR data: Begin')

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.conf = conf
        self.data_dir = _conf_get_string(conf, 'data_dir', './public_data/CASE_NAME/')
        self.image_dir = _conf_get_string(conf, 'image_dir', 'image')
        self.meta_name = _conf_get_string(conf, 'meta_name', 'radar_meta.json')
        

        self.image_paths = self._find_image_paths()
        if len(self.image_paths) == 0:
            raise FileNotFoundError('No ISAR images found in {}'.format(
                os.path.join(self.data_dir, self.image_dir)
            ))

        self.images_np = [self._load_image(path).astype(np.float32) for path in self.image_paths]
        self.n_images = len(self.images_np)
        self.image_shapes = [(image.shape[0], image.shape[1]) for image in self.images_np]
        self.H, self.W = self.image_shapes[0]
        self.image_pixels = [height * width for height, width in self.image_shapes]

        frame_dicts = self._load_frame_dicts()
        self.frame_meta_np = self._canonicalize_frame_meta(frame_dicts)

        if len(set(self.image_shapes)) == 1:
            self.images = torch.from_numpy(np.stack(self.images_np, axis=0)).to(self.device)
        else:
            self.images = [torch.from_numpy(image).to(self.device) for image in self.images_np]
            
        self.frame_meta = {
            key: torch.from_numpy(value).to(self.device)
            for key, value in self.frame_meta_np.items()
        }

        bound_radius = _conf_get_float(conf, 'bound_radius', 1.01)
        bbox_min = _conf_get(conf, 'object_bbox_min', [-bound_radius, -bound_radius, -bound_radius])
        bbox_max = _conf_get(conf, 'object_bbox_max', [bound_radius, bound_radius, bound_radius])
        self.object_bbox_min = np.asarray(bbox_min, dtype=np.float32)
        self.object_bbox_max = np.asarray(bbox_max, dtype=np.float32)

        print('Load ISAR data: End')

    def _find_image_paths(self):
        root = os.path.join(self.data_dir, self.image_dir)
        patterns = ['*.npy', '*.npz', '*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff']
        paths = []
        for pattern in patterns:
            paths.extend(glob(os.path.join(root, pattern)))
        return sorted(paths)

    def _load_image(self, path):
        ext = os.path.splitext(path)[1].lower()
        if ext == '.npy':
            image = np.load(path)
        elif ext == '.npz':
            data = np.load(path)
            key = 'image' if 'image' in data else ('isar' if 'isar' in data else data.files[0])
            image = data[key]
        else:
            image = cv.imread(path, cv.IMREAD_UNCHANGED)
            if image is None:
                raise ValueError('Failed to load image {}'.format(path))
            if image.ndim == 3:
                image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

        if np.issubdtype(image.dtype, np.integer):
            image = image.astype(np.float32) / np.iinfo(image.dtype).max
        else:
            image = np.asarray(image, dtype=np.float32)
            if image.size > 0 and np.nanmax(image) > 1.5:
                image = image / 255.0
        return image

    def _load_frame_dicts(self):
        meta_path = os.path.join(self.data_dir, self.meta_name)
        if not os.path.exists(meta_path):
            alt_ext = '.npz' if meta_path.lower().endswith('.json') else '.json'
            alt_path = os.path.splitext(meta_path)[0] + alt_ext
            if os.path.exists(alt_path):
                meta_path = alt_path
            else:
                raise FileNotFoundError('Missing per-frame radar metadata: {}'.format(meta_path))

        ext = os.path.splitext(meta_path)[1].lower()
        if ext == '.json':
            return self._load_json_frame_dicts(meta_path)
        if ext == '.npz':
            return self._load_npz_frame_dicts(meta_path)
        raise ValueError('Unsupported metadata file type: {}'.format(meta_path))

    def _load_json_frame_dicts(self, meta_path):
        with open(meta_path, 'r', encoding='utf-8') as f:
            raw = json.load(f)

        defaults = raw.get('defaults', {})
        frames = raw.get('frames', raw if isinstance(raw, list) else None)
        if frames is None:
            raise ValueError('JSON metadata must contain a "frames" list')

        if any('file' in frame for frame in frames):
            by_file = {os.path.basename(frame['file']): frame for frame in frames}
            ordered_frames = []
            for image_path in self.image_paths:
                name = os.path.basename(image_path)
                if name not in by_file:
                    raise ValueError('No metadata frame for image {}'.format(name))
                merged = dict(defaults)
                merged.update(by_file[name])
                ordered_frames.append(merged)
            return ordered_frames

        if len(frames) != len(self.image_paths):
            raise ValueError('Metadata frame count {} does not match image count {}'.format(
                len(frames), len(self.image_paths)
            ))

        ordered_frames = []
        for frame in frames:
            merged = dict(defaults)
            merged.update(frame)
            ordered_frames.append(merged)
        return ordered_frames

    def _load_npz_frame_dicts(self, meta_path):
        raw = np.load(meta_path, allow_pickle=True)
        frame_dicts = []
        for frame_idx in range(len(self.image_paths)):
            frame = {}
            for key in raw.files:
                frame[key] = _select_frame_value(raw[key], frame_idx, len(self.image_paths))
            frame_dicts.append(frame)
        return frame_dicts

    def _canonicalize_frame_meta(self, frame_dicts):
        canonical = []
        for idx, frame in enumerate(frame_dicts):
            canonical.append(self._canonicalize_one_frame(idx, frame))

        stacked = {}
        for key in canonical[0].keys():
            stacked[key] = np.stack([frame[key] for frame in canonical], axis=0).astype(np.float32)
        return stacked

    def _canonicalize_one_frame(self, frame_idx, frame):
        radar_los = _first_present(frame, _META_ALIASES['radar_los'])
        if radar_los is None:
            raise ValueError('Frame {} is missing radar_los'.format(frame_idx))
        radar_los = _normalize_vector(radar_los, 'radar_los')

        range_resolution = _first_present(frame, _META_ALIASES['range_resolution'])
        if range_resolution is None:
            raise ValueError('Frame {} is missing range_resolution'.format(frame_idx))
        range_resolution = _as_numpy(range_resolution, dtype=np.float32).reshape(-1)
        if range_resolution.shape[0] != 1:
            raise ValueError('range_resolution must be scalar, got shape {}'.format(range_resolution.shape))

        azimuth_resolution = _first_present(frame, _META_ALIASES['azimuth_resolution'])
        if azimuth_resolution is None:
            raise ValueError('Frame {} is missing azimuth_resolution'.format(frame_idx))
        azimuth_resolution = _as_numpy(azimuth_resolution, dtype=np.float32).reshape(-1)
        if azimuth_resolution.shape[0] != 1:
            raise ValueError('azimuth_resolution must be scalar, got shape {}'.format(azimuth_resolution.shape))

        range_axis = _first_present(frame, _META_ALIASES['range_axis'])
        if range_axis is None:
            raise ValueError('Frame {} is missing range_axis'.format(frame_idx))
        range_axis = _as_numpy(range_axis, dtype=np.float32).reshape(-1)

        azimuth_axis = _first_present(frame, _META_ALIASES['azimuth_axis'])
        if azimuth_axis is None:
            raise ValueError('Frame {} is missing azimuth_axis'.format(frame_idx))
        azimuth_axis = _as_numpy(azimuth_axis, dtype=np.float32).reshape(-1)

        image_size = _first_present(frame, _META_ALIASES['image_size'])
        if image_size is None:
            raise ValueError('Frame {} is missing image_size'.format(frame_idx))
        image_size = _as_numpy(image_size, dtype=np.float32).reshape(-1)
        if image_size.shape[0] != 2:
            raise ValueError('image_size must be [range, azimuth], got shape {}'.format(image_size.shape))

        image_height, image_width = self.image_shapes[frame_idx]
        expected_height, expected_width = int(image_size[0]), int(image_size[1])
        if (image_height, image_width) != (expected_height, expected_width):
            raise ValueError(
                'Frame {} image shape {} does not match image_size {}'.format(
                    frame_idx, (image_height, image_width), (expected_height, expected_width)
                )
            )
        if range_axis.shape[0] != image_height:
            raise ValueError(
                'Frame {} range_axis length {} does not match image height {}'.format(
                    frame_idx, range_axis.shape[0], image_height
                )
            )
        if azimuth_axis.shape[0] != image_width:
            raise ValueError(
                'Frame {} azimuth_axis length {} does not match image width {}'.format(
                    frame_idx, azimuth_axis.shape[0], image_width
                )
            )

        return {
            'radar_los': radar_los.astype(np.float32),
            'range_resolution': range_resolution.astype(np.float32),
            'azimuth_resolution': azimuth_resolution.astype(np.float32),
            'range_axis': range_axis.astype(np.float32),
            'azimuth_axis': azimuth_axis.astype(np.float32),
            'image_size': image_size.astype(np.float32),
        }

    def get_image_perm(self):
        return torch.randperm(self.n_images)

    def get_frame(self, frame_idx):
        return self._image_tensor(frame_idx), self.get_frame_meta(frame_idx)

    def get_frame_meta(self, frame_idx):
        return {
            key: value[frame_idx]
            for key, value in self.frame_meta.items()
        }

    def gen_random_bins_at(self, frame_idx, batch_size):
        image = self._image_tensor(frame_idx)
        height, width = image.shape
        range_idx = torch.randint(low=0, high=height, size=[batch_size], device=self.device)
        azimuth_idx = torch.randint(low=0, high=width, size=[batch_size], device=self.device)
        values = image[range_idx, azimuth_idx]
        bins = torch.stack([range_idx.float(), azimuth_idx.float()], dim=-1)
        return bins, values[:, None], self.get_frame_meta(frame_idx)

    def image_at(self, idx):
        return self._image_tensor(idx).detach().cpu().numpy()

    def _image_tensor(self, idx):
        if isinstance(self.images, list):
            return self.images[idx]
        return self.images[idx]
