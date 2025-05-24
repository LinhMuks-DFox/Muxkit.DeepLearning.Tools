# Muxkit Deep Learning Tools

This repository provides reusable utilities for audio and vision research.

## audio_tools

### bc_augmentation
- `mix_sounds(sound1, sound2, r, fs, device="cpu")` – mix two audio tensors with perceptual gain compensation.
- `BCAugmentor(sample_rate)` – module wrapper exposing `.mix_sounds()`.
- `BCLearningDataset(dataset, sample_rate, num_classes, device='cpu')` – dataset applying Between-Class augmentation.

### transforms
- `create_mask(size, mask_rate=0.5)` – generate a boolean mask tensor.
- `tensor_masking(tensor, mask_rate=0.5, mask=None)` – return `(masked, unmasked, mask)`.
- `TimeSequenceLengthFixer(fixed_length, sample_rate, mode="r")` – trim or pad audio to a fixed duration.
- `SoundTrackSelector(mode)` – select left/right/mixed channel from stereo audio.
- `TimeSequenceMaskingTransformer(mask_rate)` – time-domain masking module.
- `SpectrogramMaskingTransformer(mask_rate)` – 2‑D spectrogram masking module.

### utils
- `fix_length(audio_data, sample_length)` – pad or crop audio to `sample_length` samples.

## dataset_tools
- `CacheableDataset(dataset, max_cache_size=1000, multiprocessing=False, device='cpu')` – cache dataset items in memory.
- `label_digit2tensor(digits, class_num)` – convert label indices to a one-hot tensor.

## lossfunction
- `PLMLoss(label_distribution, lambda_rate=0.1, hist_bins=10, loss_kernel="bce")` – partial label masking loss. Call `update_ratios()` periodically.

## model_tools
- `freeze_module(module)` / `unfreeze_module(module)` – toggle gradient computation for all parameters.
- `stati_model(model, unit='bytes')` – return parameter counts and approximate memory footprint.

## plot_tools
- `plot_confusion_matrix(matrix)` – draw a normalized confusion matrix with counts.
- `ConfusionMatrixPlotter(class2label)` – `plot(confusion_matrix, n_rows=1, n_cols=1)` handles multi-class and multi-label matrices.

## score_tools
- `MonoLabelClassificationTester(model, device, loss_fn)` provides evaluation helpers:
  - `set_dataloader(dataloader, n_class)`
  - `predict_all()`
  - `calculate_all_metrics()`
  - `status_map()` → dictionary with F1, accuracy, precision and recall.

## train_tools
- `set_manual_seed(seed)` – set deterministic seeds for PyTorch, NumPy and Python random.

## utl.api_tags
Decorators to tag API status: `stable_api`, `deprecated`, `unfinished_api`, `untested`, `bug_api`.

Unit tests live under each module's `unit_test` directory. Dependencies are declared in `pyproject.toml`.
