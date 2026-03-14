"""LeRobot dataset utilities with standardized key conventions.

This module provides utilities for loading LeRobot datasets with proper
observation history and action horizon configuration.

Key Convention:
    All datasets are expected to use the following key format.
    Note: LeRobot doesn't allow '/' in feature names, so we use '.' as separator.

    Observations (prefixed with "observation."):
        - observation.image: Base camera image (H, W, C)
        - observation.wrist_image: Wrist camera image (H, W, C)
        - observation.state: Robot proprioceptive state (D,)

    Actions:
        - actions: Action sequence (horizon, action_dim)

    Metadata (auto-excluded):
        - timestamp, frame_index, episode_index, index, task_index

    This convention aligns with the adaptor expected format in openpi.adaptors.
"""

import logging
import os

import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm
import hydra

logger = logging.getLogger(__name__)

from lerobot.datasets.lerobot_dataset import (
    LeRobotDataset,
    LeRobotDatasetMetadata,
)


# Monkey-patch LeRobot v3.0's slow _query_hf_dataset
# Uses select() instead of direct indexing for ~40x speedup
# See: https://github.com/huggingface/lerobot/issues/2895
def _fast_query_hf_dataset(self, query_indices: dict[str, list[int]]) -> dict:
    size = len(self.hf_dataset)
    return {
        key: torch.stack(list(self.hf_dataset.select([max(0, min(i, size - 1)) for i in q_idx])[key]))
        for key, q_idx in query_indices.items()
        if key not in self.meta.video_keys
    }


LeRobotDataset._query_hf_dataset = _fast_query_hf_dataset



class TransformedDataset(torch.utils.data.Dataset):
    """Dataset wrapper that applies adaptor transform to each sample.

    Loads adaptor from config internally - fully self-contained.
    The adaptor handles norm_stats loading and tokenizer creation internally.
    """

    def __init__(self, dataset, adaptor_cfg):
        """
        Args:
            dataset: Base dataset (e.g., LeRobotDataset)
            adaptor_cfg: OmegaConf dict with _target_ and adaptor parameters.
                        Must include norm_stats_path and optionally max_token_len.
                        Pass None to disable transformation.

                        Config should use nested format (Adaptor with robot: and model: sub-configs).
        """
        self._dataset = dataset
        self._adaptor = None

        if adaptor_cfg is not None:
            # Build override kwargs for instantiation
            override_kwargs = {}

            # Extract tasks from dataset metadata if available (for prompt lookup)
            tasks_dict = self._extract_tasks(dataset)

            # Detect new vs old config format by checking for 'robot' key
            has_robot_key = OmegaConf.select(adaptor_cfg, "robot") is not None

            if has_robot_key and tasks_dict is not None:
                # New composable format: inject tasks into robot sub-config
                if not OmegaConf.select(adaptor_cfg, "robot.tasks"):
                    robot_cfg = OmegaConf.select(adaptor_cfg, "robot")
                    if robot_cfg is not None:
                        override_kwargs["robot"] = hydra.utils.instantiate(
                            robot_cfg, tasks=tasks_dict
                        )
                    logger.info(f"TransformedDataset: injected {len(tasks_dict)} tasks into robot config")
            elif tasks_dict is not None:
                # Legacy format: inject tasks at top level
                if not OmegaConf.select(adaptor_cfg, "tasks"):
                    override_kwargs["tasks"] = tasks_dict
                    logger.info(f"TransformedDataset: loaded {len(tasks_dict)} tasks from dataset metadata")

            self._adaptor = hydra.utils.instantiate(adaptor_cfg, **override_kwargs)
            logger.info(f"TransformedDataset using adaptor: {type(self._adaptor).__name__}")

            # Auto-compute norm stats if requested (norm_stats_path="auto")
            model = getattr(self._adaptor, "model", None)
            if model is not None and getattr(model, "_auto_norm_stats", False):
                model.auto_compute_norm_stats(dataset.hf_dataset)
                # Update parent adaptor's cached reference
                if hasattr(self._adaptor, "_norm_stats"):
                    self._adaptor._norm_stats = model.get_norm_stats()

    @staticmethod
    def _extract_tasks(dataset) -> dict[int, str] | None:
        """Extract task mapping from dataset metadata."""
        if not hasattr(dataset, "meta") or not hasattr(dataset.meta, "tasks"):
            return None

        tasks = dataset.meta.tasks
        tasks_not_empty = tasks is not None and (not hasattr(tasks, "empty") or not tasks.empty)
        if not tasks_not_empty:
            return None

        if hasattr(tasks, "iterrows"):
            return {int(row["task_index"]): task_str for task_str, row in tasks.iterrows()}
        return tasks

    def __getitem__(self, index):
        sample = self._dataset[index]
        if self._adaptor is not None:
            sample = self._adaptor.datasets_input_transforms(sample)
        return sample

    def __len__(self):
        return len(self._dataset)

    def __getattr__(self, name):
        # Guard against infinite recursion during unpickling:
        # When pickle restores the object, __dict__ is empty so accessing
        # self._dataset triggers __getattr__ again → RecursionError.
        if name == "_dataset":
            raise AttributeError(name)
        return getattr(self._dataset, name)

    @property
    def adaptor(self):
        """Access the adaptor instance (useful for getting dimensions etc.)"""
        return self._adaptor

    def print_sample_format(self, index: int = 0) -> None:
        """Print shape, dtype, and value range of a raw and transformed sample."""
        self._print_format(self._dataset, index=index, tag="RAW DATASET")
        self._print_format(self, index=index, tag="AFTER ADAPTOR")

    @staticmethod
    def _print_format(dataset, index: int = 0, tag: str = "DATASET SAMPLE") -> None:
        """Print shape, dtype, and value range of a dataset sample.

        Args:
            dataset: Any dataset with __getitem__.
            index: Sample index to inspect.
            tag: Label for the printout.
        """
        import numpy as np

        sample = dataset[index]
        lines = [f"[{tag}] index={index}:"]
        for key in sorted(sample.keys()):
            val = sample[key]
            if isinstance(val, torch.Tensor):
                lines.append(
                    f"  {key}: shape={list(val.shape)} dtype={val.dtype} "
                    f"range=[{val.min().item():.4f}, {val.max().item():.4f}]"
                )
            elif isinstance(val, np.ndarray):
                lines.append(
                    f"  {key}: shape={list(val.shape)} dtype={val.dtype} "
                    f"range=[{val.min():.4f}, {val.max():.4f}]"
                )
            elif isinstance(val, dict):
                lines.append(f"  {key}: dict with {len(val)} keys")
                for sub_key in sorted(val.keys()):
                    sub_val = val[sub_key]
                    if isinstance(sub_val, (torch.Tensor, np.ndarray)):
                        mn = sub_val.min().item() if isinstance(sub_val, torch.Tensor) else sub_val.min()
                        mx = sub_val.max().item() if isinstance(sub_val, torch.Tensor) else sub_val.max()
                        lines.append(
                            f"    {sub_key}: shape={list(sub_val.shape)} dtype={sub_val.dtype} "
                            f"range=[{mn:.4f}, {mx:.4f}]"
                        )
                    else:
                        lines.append(f"    {sub_key}: type={type(sub_val).__name__} value={sub_val!r}")
            else:
                lines.append(f"  {key}: type={type(val).__name__} value={val!r}")
        print("\n".join(lines), flush=True)


class InMemoryTransformedDataset(torch.utils.data.Dataset):
    """TransformedDataset that preloads all samples into memory.

    Same interface as TransformedDataset but precomputes all adaptor
    transforms at init and stores results as contiguous tensors.
    Orders of magnitude faster for low-dim-only datasets.
    """

    def __init__(self, dataset, adaptor_cfg, cache_path=None):
        # Reuse TransformedDataset for adaptor setup (norm stats, etc.)
        wrapped = TransformedDataset(dataset, adaptor_cfg)
        self._adaptor = wrapped.adaptor

        # Try loading from disk cache
        if cache_path and os.path.exists(cache_path):
            logger.info(f"Loading preloaded data from cache: {cache_path}")
            self._data = torch.load(cache_path, weights_only=False)
            self._log_summary(len(self))
            return

        n = len(dataset)
        logger.info(f"Preloading {n} samples into memory...")

        # Speed up: load HF columns into tensors so _query_hf_dataset
        # uses tensor indexing instead of slow hf_dataset.select()
        base_ds = wrapped._dataset
        if hasattr(base_ds, "hf_dataset"):
            raw_samples = self._preload_fast(wrapped, base_ds, n)
        else:
            raw_samples = [wrapped[i] for i in tqdm(range(n), desc="Preloading")]

        # Stack into contiguous tensors
        self._data = self._stack_samples(raw_samples)
        self._log_summary(n)

        # Save to disk cache for next run
        if cache_path:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            logger.info(f"Saving preloaded data to cache: {cache_path}")
            torch.save(self._data, cache_path)

    def _log_summary(self, n):
        keys_summary = []
        for k, v in self._data.items():
            if isinstance(v, torch.Tensor):
                keys_summary.append(f"{k}: {list(v.shape)}")
            elif isinstance(v, dict):
                sub = {sk: list(sv.shape) for sk, sv in v.items() if isinstance(sv, torch.Tensor)}
                keys_summary.append(f"{k}: {sub}")
        logger.info(f"Preloaded {n} samples: {', '.join(keys_summary)}")

    @staticmethod
    def _preload_fast(wrapped, base_ds, n):
        """Preload by caching HF columns as tensors for fast indexing.

        Temporarily patches _query_hf_dataset on the dataset instance so
        that temporal stacking uses tensor fancy indexing instead of
        hf_dataset.select() (which does parquet I/O per call).
        """
        hf = base_ds.hf_dataset
        video_keys = base_ds.meta.video_keys if hasattr(base_ds, "meta") else set()

        # Load all numeric columns into memory as tensors
        logger.info(f"Loading {len(hf.column_names)} HF columns into memory...")
        mem_cols = {}
        for col in hf.column_names:
            try:
                mem_cols[col] = torch.as_tensor(np.array(hf[col]))
            except (ValueError, TypeError):
                logger.debug(f"Skipping non-numeric column: {col}")

        # Patch instance to use in-memory tensor indexing
        def _mem_query(query_indices):
            return {
                key: mem_cols[key][q_idx]
                for key, q_idx in query_indices.items()
                if key not in video_keys and key in mem_cols
            }

        base_ds._query_hf_dataset = _mem_query

        try:
            samples = [wrapped[i] for i in tqdm(range(n), desc="Preloading")]
        finally:
            # Remove instance override — falls back to class-level method
            del base_ds._query_hf_dataset
            del mem_cols

        return samples

    @staticmethod
    def _stack_samples(samples):
        """Stack list of sample dicts into a single dict of tensors."""
        first = samples[0]
        result = {}
        for key in first:
            vals = [s[key] for s in samples]
            if isinstance(vals[0], dict):
                sub_keys = vals[0].keys()
                result[key] = {
                    sk: torch.stack([torch.as_tensor(v[sk]) for v in vals])
                    for sk in sub_keys
                }
            elif isinstance(vals[0], (np.ndarray, torch.Tensor)):
                result[key] = torch.stack([torch.as_tensor(v) for v in vals])
            else:
                result[key] = vals  # strings etc — keep as list
        return result

    def __getitem__(self, idx):
        result = {}
        for key, val in self._data.items():
            if isinstance(val, dict):
                result[key] = {k: v[idx] for k, v in val.items()}
            elif isinstance(val, torch.Tensor):
                result[key] = val[idx]
            else:
                result[key] = val[idx]
        return result

    def __len__(self):
        for val in self._data.values():
            if isinstance(val, torch.Tensor):
                return val.shape[0]
            if isinstance(val, dict):
                for v in val.values():
                    if isinstance(v, torch.Tensor):
                        return v.shape[0]
        return 0

    @property
    def adaptor(self):
        return self._adaptor


class _InMemoryHFDataset:
    """Lightweight in-memory replacement for HF Dataset.

    Supports the subset of HF Dataset API used by LeRobotDataset:
    - __getitem__(int) → dict of tensors (single row)
    - __getitem__(str) → full column as tensor or list
    - __len__() → int
    - column_names → list[str]
    - select(indices) → view with [key] → list of tensors
    - unique(column) → sorted unique values
    """

    def __init__(self, numeric_cols: dict[str, torch.Tensor], string_cols: dict[str, list], size: int):
        self._numeric = numeric_cols
        self._string = string_cols
        self._size = size
        self.column_names = list(numeric_cols.keys()) + list(string_cols.keys())

    def __len__(self):
        return self._size

    def __getitem__(self, key):
        if isinstance(key, int):
            item = {}
            for k, v in self._numeric.items():
                item[k] = v[key]
            for k, v in self._string.items():
                item[k] = v[key]
            return item
        elif isinstance(key, str):
            if key in self._numeric:
                return self._numeric[key]
            if key in self._string:
                return self._string[key]
            raise KeyError(key)
        raise TypeError(f"Unsupported key type: {type(key)}")

    def select(self, indices):
        """Return a view for the given indices (HF Dataset compatibility)."""
        return _InMemoryHFView(self, indices)

    def unique(self, column: str):
        """Return unique values for a column."""
        if column in self._numeric:
            return self._numeric[column].unique().tolist()
        if column in self._string:
            return sorted(set(self._string[column]))
        raise KeyError(column)


class _InMemoryHFView:
    """View into _InMemoryHFDataset for select() compatibility."""

    def __init__(self, parent: _InMemoryHFDataset, indices: list[int]):
        self._parent = parent
        self._indices = indices

    def __getitem__(self, key: str):
        if key in self._parent._numeric:
            return [self._parent._numeric[key][i] for i in self._indices]
        if key in self._parent._string:
            return [self._parent._string[key][i] for i in self._indices]
        raise KeyError(key)

    def __len__(self):
        return len(self._indices)


def cache_dataset_in_memory(dataset) -> None:
    """Cache all HF dataset columns in memory for fast random access.

    Replaces slow parquet I/O in __getitem__ and _query_hf_dataset with
    in-memory tensor indexing. Suitable for low-dim datasets (no images).

    After calling this function:
    - dataset.hf_dataset is replaced with an in-memory wrapper
    - dataset._query_hf_dataset uses tensor fancy indexing
    - Original parquet-backed HF dataset is released

    Memory: ~7GB for 25M-frame DROID low-dim dataset.
    Tensors are placed in shared memory for efficient multi-worker DataLoader.

    Loads data directly from parquet files using pyarrow for fast bulk I/O
    (~60s for 25M frames vs hours through HF Dataset).

    Args:
        dataset: A LeRobotDataset instance with hf_dataset loaded.
    """
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    n = len(dataset.hf_dataset)
    video_keys = set(dataset.meta.video_keys) if hasattr(dataset, "meta") else set()

    # Read all parquet files in one pass
    data_dir = dataset.root / "data"
    parquet_files = sorted(data_dir.rglob("*.parquet"))
    logger.info(f"Caching {n:,} frames from {len(parquet_files)} parquet files...")

    tables = [pq.read_table(f) for f in tqdm(parquet_files, desc="Reading parquet")]
    full_table = pa.concat_tables(tables)
    del tables

    # Convert Arrow columns to tensors
    numeric_cols = {}
    string_cols = {}

    for col in tqdm(full_table.column_names, desc="Converting columns"):
        if col in video_keys:
            continue
        arrow_col = full_table.column(col)
        col_type = arrow_col.type

        if pa.types.is_string(col_type) or pa.types.is_large_string(col_type):
            string_cols[col] = arrow_col.to_pylist()
        elif pa.types.is_list(col_type) or pa.types.is_large_list(col_type):
            # list<double> → 2D tensor via list_flatten + reshape (17x faster than np.stack)
            flat = pc.list_flatten(arrow_col)
            arr = flat.to_numpy(zero_copy_only=False).astype(np.float32).reshape(n, -1)
            tensor = torch.from_numpy(arr.copy())
            tensor.share_memory_()
            numeric_cols[col] = tensor
        else:
            # Scalar columns (int64, float64, bool)
            arr = arrow_col.to_numpy(zero_copy_only=False)
            tensor = torch.as_tensor(arr.copy() if not arr.flags.writeable else arr)
            tensor.share_memory_()
            numeric_cols[col] = tensor

    del full_table

    mem_bytes = sum(t.nelement() * t.element_size() for t in numeric_cols.values())
    logger.info(
        f"Cached {len(numeric_cols)} numeric + {len(string_cols)} string columns "
        f"({mem_bytes / 1024**3:.2f} GB)"
    )

    # Replace hf_dataset with lightweight wrapper
    dataset.hf_dataset = _InMemoryHFDataset(numeric_cols, string_cols, n)

    # Override _query_hf_dataset on instance for fast tensor fancy indexing
    def _mem_query(query_indices):
        return {
            key: numeric_cols[key][torch.clamp(
                torch.tensor(q_idx, dtype=torch.long), 0, n - 1
            )]
            for key, q_idx in query_indices.items()
            if key not in video_keys and key in numeric_cols
        }

    dataset._query_hf_dataset = _mem_query


def create_lerobot_dataset(
    repo_id: str,
    n_horizon: int = 50,
    n_obs: int = 1,
    n_latency: int = 0,
    obs_keys: list[str] | None = None,
    action_keys: list[str] | None = None,
) -> LeRobotDataset:
    """
    Create a LeRobotDataset with configured observation history and action horizon.

    By default, auto-detects observation and action keys from dataset metadata.
    Use obs_keys and action_keys to override with specific keys.

    Args:
        repo_id: Dataset identifier (e.g., "lerobot/aloha_sim_transfer_cube_human").
                 Dataset is located at $HF_LEROBOT_HOME/{repo_id}
        n_horizon: Number of future action frames to include
        n_obs: Number of observation frames (1 = current only, 2 = current + 1 past, etc.)
        n_latency: Action latency in frames (0 = actions start at t, 1 = actions start at t+1)
        obs_keys: List of observation keys to apply timestamps to (default: auto-detect)
        action_keys: List of action keys to apply timestamps to (default: auto-detect)

    Returns:
        LeRobotDataset with delta_timestamps configured for specified keys

    Example:
        # Auto-detect all keys
        dataset = create_lerobot_dataset("lerobot/pusht", n_horizon=50, n_obs=1)

        # Specify only certain observation keys
        dataset = create_lerobot_dataset(
            "lerobot/aloha_sim_transfer_cube_human",
            n_horizon=50,
            n_obs=2,
            obs_keys=["observation.image", "observation.state"],
            action_keys=["actions"],
        )

        # With latency: actions start 1 frame after observation
        dataset = create_lerobot_dataset("lerobot/pusht", n_horizon=50, n_obs=1, n_latency=1)
    """
    # Get metadata first (lightweight, doesn't load full dataset)
    dataset_meta = LeRobotDatasetMetadata(repo_id)
    fps = dataset_meta.fps
    features = dataset_meta.features

    # Build observation timestamps: [-(n_obs-1)/fps, ..., -1/fps, 0.0]
    # e.g., n_obs=3 -> [-0.2, -0.1, 0.0] at 10fps
    obs_timestamps = [i / fps for i in range(-(n_obs - 1), 1)]

    # Build action timestamps aligned with obs start (matching DecoupledActionHead's pad_before behavior):
    # Actions start at the same frame as the first obs frame.
    # e.g., n_obs=2, n_horizon=16, n_latency=0 -> [-0.05, 0.0, 0.05, ..., 0.70] at 20fps
    action_start = -(n_obs - 1) + n_latency
    action_timestamps = [(action_start + i) / fps for i in range(n_horizon)]

    # Build delta_timestamps dict
    delta_timestamps = {}

    # If keys are explicitly provided, use them directly
    if obs_keys is not None and action_keys is not None:
        for key in obs_keys:
            delta_timestamps[key] = obs_timestamps
        for key in action_keys:
            delta_timestamps[key] = action_timestamps
    else:
        # Auto-detect from features
        # Expected key format:
        #   - Observations: "observation.image", "observation.wrist_image", "observation.state"
        #   - Actions: "actions"

        # Metadata keys to skip
        skip_keys = {"timestamp", "frame_index", "episode_index", "index", "task_index"}

        detected_obs_keys = []
        detected_action_keys = []

        for key in features:
            if key in skip_keys:
                continue

            # Action keys (singular or plural)
            if key in ("action", "actions"):
                detected_action_keys.append(key)
            # Observation keys: all keys starting with "observation."
            elif key.startswith("observation."):
                detected_obs_keys.append(key)

        # Use provided keys if specified, otherwise use detected
        final_obs_keys = obs_keys if obs_keys is not None else detected_obs_keys
        final_action_keys = action_keys if action_keys is not None else detected_action_keys

        for key in final_obs_keys:
            delta_timestamps[key] = obs_timestamps
        for key in final_action_keys:
            delta_timestamps[key] = action_timestamps

    # Create dataset with delta_timestamps
    dataset = LeRobotDataset(repo_id=repo_id, delta_timestamps=delta_timestamps)

    # Force HF dataset load and fix metadata if total_frames is wrong.
    # DROID metadata total_frames > actual HF rows, causing DataLoader IndexError.
    # num_frames returns meta.total_frames when episodes=None, so we must patch it.
    dataset._ensure_hf_dataset_loaded()
    actual_size = len(dataset.hf_dataset)
    if dataset.meta.total_frames != actual_size:
        logger.warning(
            f"Metadata total_frames ({dataset.meta.total_frames}) != "
            f"actual HF dataset size ({actual_size}). Patching metadata."
        )
        dataset.meta.total_frames = actual_size

    # Skip video decoding if no video keys are requested in delta_timestamps
    if dataset.meta.video_keys:
        needed_video_keys = set(delta_timestamps.keys()) & set(dataset.meta.video_keys)
        if not needed_video_keys:
            skipped = list(dataset.meta.video_keys)
            for vk in skipped:
                dataset.meta.features[vk]["dtype"] = "video_disabled"
            logger.info(f"Disabled video decoding for {skipped} (not in delta_timestamps)")

    # Print summary
    summary_obs_keys = [k for k in delta_timestamps if k.startswith("observation.")]
    action_key = "actions" if "actions" in delta_timestamps else "action"

    # Get basic info from metadata
    info = dataset_meta.info
    codebase_version = info.get("codebase_version", "unknown")
    robot_type = info.get("robot_type", "unknown")
    total_tasks = info.get("total_tasks", 0)

    logger.info("=" * 70)
    logger.info("LEROBOT DATASET")
    logger.info("=" * 70)
    logger.info(f"  repo_id:    {repo_id}")
    logger.info(f"  version:    {codebase_version}")
    logger.info(f"  robot:      {robot_type}")
    logger.info(f"  frames:     {dataset_meta.total_frames:,}")
    logger.info(f"  episodes:   {dataset_meta.total_episodes:,}")
    logger.info(f"  tasks:      {total_tasks}")
    logger.info(f"  fps:        {fps}")
    logger.info(f"  n_obs:      {n_obs} frames ({n_obs / fps:.2f}s history)")
    logger.info(f"  n_horizon:  {n_horizon} frames ({n_horizon / fps:.2f}s future)")
    logger.info(f"  n_latency:  {n_latency} frames ({n_latency / fps:.2f}s delay)")
    logger.info(f"  obs_keys:   {summary_obs_keys}")
    logger.info(f"  action_key: {action_key}")
    logger.info("=" * 70)

    return dataset


if __name__ == "__main__":
    # Test on libero LeRobot dataset
    # Assumes HF_LEROBOT_HOME is set appropriately
    # Dataset keys expected: observation.image, observation.wrist_image, observation.state, actions
    REPO_ID = "JianZhou0420/libero_openvla_LeRobotv3_0"

    print("=" * 60)
    print("Testing create_lerobot_dataset")
    print("=" * 60)

    # Test 1: Pi0 style (1 obs, 50 horizon)
    print("\n[Test 1] Pi0 style: n_obs=1, n_horizon=50, n_latency=0")
    dataset = create_lerobot_dataset(
        repo_id=REPO_ID,
        n_horizon=50,
        n_obs=1,
        n_latency=0,
    )
    print(f"  Dataset size: {len(dataset)}")
    print(f"  FPS: {dataset.fps}")
    print(f"  delta_timestamps keys: {list(dataset.delta_timestamps.keys())}")

    sample = dataset[100]
    print(f"  Sample keys: {list(sample.keys())}")
    for key, value in sample.items():
        if hasattr(value, "shape"):
            print(f"    {key}: {value.shape}")

    # Test 2: Diffusion Policy style (2 obs history, 16 horizon)
    print("\n[Test 2] Diffusion Policy style: n_obs=2, n_horizon=16, n_latency=0")
    dataset2 = create_lerobot_dataset(
        repo_id=REPO_ID,
        n_horizon=16,
        n_obs=2,
        n_latency=0,
    )
    sample2 = dataset2[100]
    for key, value in sample2.items():
        if hasattr(value, "shape"):
            print(f"    {key}: {value.shape}")

    # Test 3: With latency
    print("\n[Test 3] With latency: n_obs=1, n_horizon=50, n_latency=1")
    dataset3 = create_lerobot_dataset(
        repo_id=REPO_ID,
        n_horizon=50,
        n_obs=1,
        n_latency=1,
    )
    sample3 = dataset3[100]
    # Find the action key (could be 'action' or 'actions')
    action_key = "actions" if "actions" in dataset3.delta_timestamps else "action"
    print(f"  Action timestamps start: {dataset3.delta_timestamps[action_key][:3]}")
    print(f"  Action timestamps end: {dataset3.delta_timestamps[action_key][-3:]}")

    # Test 4: DataLoader
    print("\n[Test 4] DataLoader test")
    from torch.utils.data import DataLoader

    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)
    batch = next(iter(dataloader))
    print(f"  Batch keys: {list(batch.keys())}")
    for key, value in batch.items():
        if hasattr(value, "shape"):
            print(f"    {key}: {value.shape}")

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
