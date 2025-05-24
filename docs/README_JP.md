# Muxkit ディープラーニングツール

このリポジトリは、音声および画像研究で再利用できるユーティリティ群を提供します。

## audio_tools

### bc_augmentation
- `mix_sounds(sound1, sound2, r, fs, device="cpu")`：二つの音声を知覚的なゲイン調整付きで混合します。
- `BCAugmentor(sample_rate)`：`mix_sounds` をラップしたモジュールです。
- `BCLearningDataset(dataset, sample_rate, num_classes, device='cpu')`：BC増強を適用するデータセット。

### transforms
- `create_mask(size, mask_rate=0.5)`：ブールマスクを生成します。
- `tensor_masking(tensor, mask_rate=0.5, mask=None)`：`(masked, unmasked, mask)` を返します。
- `TimeSequenceLengthFixer(fixed_length, sample_rate, mode="r")`：音声を指定時間に切り詰めまたはパディングします。
- `SoundTrackSelector(mode)`：ステレオから左・右・ミックスを選択します。
- `TimeSequenceMaskingTransformer(mask_rate)`：時系列マスキングのモジュール。
- `SpectrogramMaskingTransformer(mask_rate)`：スペクトログラムを2次元でマスキングします。

### utils
- `fix_length(audio_data, sample_length)`：音声を指定サンプル数に調整します。

## dataset_tools
- `CacheableDataset(dataset, max_cache_size=1000, multiprocessing=False, device='cpu')`：データをメモリにキャッシュします。
- `label_digit2tensor(digits, class_num)`：ラベルインデックスを one-hot テンソルに変換します。

## lossfunction
- `PLMLoss(label_distribution, lambda_rate=0.1, hist_bins=10, loss_kernel="bce")`：部分ラベルマスキング損失。定期的に `update_ratios()` を呼び出してください。

## model_tools
- `freeze_module(module)` / `unfreeze_module(module)`：モジュールのパラメータの勾配計算を切り替えます。
- `stati_model(model, unit='bytes')`：パラメータ数とおおよそのメモリ使用量を返します。

## plot_tools
- `plot_confusion_matrix(matrix)`：正規化された混同行列を描画します。
- `ConfusionMatrixPlotter(class2label)`：`plot(confusion_matrix, n_rows=1, n_cols=1)` で多クラス・多ラベルに対応。

## score_tools
- `MonoLabelClassificationTester(model, device, loss_fn)` により評価を行います：
  - `set_dataloader(dataloader, n_class)`
  - `predict_all()`
  - `calculate_all_metrics()`
  - `status_map()` で F1、Accuracy、Precision、Recall を取得。

## train_tools
- `set_manual_seed(seed)`：PyTorch、NumPy、Python の乱数シードを一括設定します。

## utl.api_tags
APIの状態を示すデコレータ：`stable_api`、`deprecated`、`unfinished_api`、`untested`、`bug_api`。

各モジュールの `unit_test` ディレクトリにテストがあり、依存関係は `pyproject.toml` で管理されています。
