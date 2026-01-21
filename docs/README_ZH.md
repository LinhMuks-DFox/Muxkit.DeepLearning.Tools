# Muxkit 深度学习工具集

该仓库提供在音频和视觉研究中可复用的工具代码。

## 数据集（datasets）
- 下载策略：优先使用外部工具（curl/wget）。当系统无外部工具时，显式传入 `by_internal_downloader=True` 使用基于 requests 的内部下载，`headers` 可自定义 HTTP 头。
- 依赖安装（用于数据集下载与可视化）：
  - pip：`pip install ".[datasets]"`
  - Poetry：`poetry install -E datasets`

示例
- HOMULA‑RIR：
  - 外部：`HomulaRIR.download("./data/HOMULA-RIR")`
  - 内部：`HomulaRIR.download("./data/HOMULA-RIR", by_internal_downloader=True, headers={"User-Agent": "..."})`
- JVS（需要 gdown）：
  - `JVSDataset.download("./data/JVS-org/jvs_ver1/")` 或 `JVSDataset.download_with_gdown("./data/JVS-org/jvs_ver1/")`

## audio_tools

### bc_augmentation
- `mix_sounds(sound1, sound2, r, fs, device="cpu")`：按感知增益混合两段音频。
- `BCAugmentor(sample_rate)`：封装 `mix_sounds` 的模块。
- `BCLearningDataset(dataset, sample_rate, num_classes, device='cpu')`：在数据集中应用 BC 增强。

### transforms
- `create_mask(size, mask_rate=0.5)`：生成布尔掩码张量。
- `tensor_masking(tensor, mask_rate=0.5, mask=None)`：返回 `(masked, unmasked, mask)`。
- `TimeSequenceLengthFixer(fixed_length, sample_rate, mode="r")`：将音频裁剪或填充到固定时长。
- `SoundTrackSelector(mode)`：从立体声中选择左声道/右声道或混合通道。
- `TimeSequenceMaskingTransformer(mask_rate)`：时域掩码模块。
- `SpectrogramMaskingTransformer(mask_rate)`：二维谱图掩码模块。

### utils
- `fix_length(audio_data, sample_length)`：将音频补齐或截断到给定采样数。

## dataset_tools
- `CacheableDataset(dataset, max_cache_size=1000, multiprocessing=False, device='cpu')`：在内存中缓存数据样本。
- `label_digit2tensor(digits, class_num)`：将标签索引列表转成 one-hot 张量。

## lossfunction
- `PLMLoss(label_distribution, lambda_rate=0.1, hist_bins=10, loss_kernel="bce")`：部分标签掩码损失，可定期调用 `update_ratios()` 更新比例。

## model_tools
- `freeze_module(module)` / `unfreeze_module(module)`：冻结或解冻模型参数。
- `stati_model(model, unit='bytes')`：获取参数数量及大致占用内存。

## plot_tools
- `plot_confusion_matrix(matrix)`：绘制归一化并带计数的混淆矩阵。
- `ConfusionMatrixPlotter(class2label)`：`plot(confusion_matrix, n_rows=1, n_cols=1)` 处理多类或多标签矩阵。

## score_tools
- `MonoLabelClassificationTester(model, device, loss_fn)` 提供评估流程：
  - `set_dataloader(dataloader, n_class)`
  - `predict_all()`
  - `calculate_all_metrics()`
  - `status_map()` 返回 F1、准确率、精确率和召回率。

## train_tools
- `set_manual_seed(seed)`：同时设置 PyTorch、NumPy 和随机库的种子。

## utl.api_tags
API 状态装饰器：`stable_api`、`deprecated`、`unfinished_api`、`untested`、`bug_api`。

各模块在 `unit_test` 目录下提供测试示例，依赖项见 `pyproject.toml`。
