"""
HOMULA-RIR dataset access utilities and Dataset wrapper.

Reference
- "A Room Impulse Response Dataset for Teleconferencing and Spatial Audio Applications Acquired Through Higher-Order Microphones and Uniform Linear Microphone Arrays" (arXiv:2402.13896)
"""

import os
import glob
import torch
import torchaudio
import pandas as pd
import typing
import zipfile
import matplotlib.pyplot as plt
from .base import DownloadMixin, DownloadError


class HomulaRIR(torch.utils.data.Dataset, DownloadMixin):

    LINK = "https://zenodo.org/records/10479726/files/HOMULA-RIR.zip?download=1"
    ZIP_NAME = "HOMULA-RIR.zip"

    ROWS = range(1, 6)
    COLS = range(1, 6)
    SOURCES = ["S1", "S2"]  # 两个声源

    def __init__(self, root_dir: str, download: bool = False):
        """Initialize the dataset.

        Args:
            root_dir (str): Root folder containing 'hom' and 'ula' subfolders.
            download (bool): If True and missing, download the archive.
        """
        self.root_dir = root_dir

        if download:
            self.download(root_dir)

        if not self._check_integrity():
            raise RuntimeError(
                f"Dataset not found or corrupted at {root_dir}. Use download=True to download it.")

        self.samples = []
        for r in self.ROWS:
            for c in self.COLS:
                for s in self.SOURCES:
                    self.samples.append({
                        "row": r,
                        "col": c,
                        "source": s
                    })

    def _check_integrity(self) -> bool:
        return os.path.isdir(os.path.join(self.root_dir, "hom")) and \
            os.path.isdir(os.path.join(self.root_dir, "ula"))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        meta = self.samples[idx]
        return self.get_high_order_microphone(meta["row"], meta["col"], meta["source"])

    def get_high_order_microphone(self, row: int, col: int, source: str = "S1") -> dict:
        """Load HOM (higher-order microphone) sample at grid position.

        Args:
            row (int): 1–5 grid row.
            col (int): 1–5 grid column.
            source (str): 'S1' or 'S2'.
        """
        # 构建路径
        # 结构: hom/row{r}/rir-{s}-R{r}-HOM{c}.wav
        wav_filename = f"rir-{source}-R{row}-HOM{col}.wav"
        pos_filename = f"pos-R{row}-HOM{col}.csv"

        wav_path = os.path.join(self.root_dir, "hom",
                                f"row{row}", wav_filename)
        pos_path = os.path.join(self.root_dir, "hom",
                                f"row{row}", pos_filename)

        # 读取音频 (Tensor: channels x time, sample_rate)
        waveform, sr = torchaudio.load(wav_path)

        # 读取坐标 (使用 pandas 读取 csv)
        # CSV通常包含 x,y,z
        df_pos = pd.read_csv(pos_path)
        coords = torch.tensor(df_pos.values, dtype=torch.float32)

        return {
            "type": "HOM",
            "wav": waveform,      # Shape: [Channels, Time]
            "sample_rate": sr,
            "coordinates": coords,  # Shape: [1, 3] usually
            "meta": {"row": row, "col": col, "source": source}
        }

    def get_uniform_linear_array(self, source: str = "S1") -> dict:
        """Load ULA (64-ch) data. ULA is outside the grid and fetched separately."""
        filename = f"rir_{source}-ULA.wav" if source == "S2" else f"rir-{source}-ULA.wav"  # 注意S2文件名里可能有下划线差异，需根据实际文件微调，这里假设格式统一
        # 根据你的 tree，S2 是 rir_S2-ULA.wav (下划线)，S1 是 rir-S1-ULA.wav (连字符)
        # 这是一个常见的数据集命名坑，这里做个自动处理：
        if source == "S2":
            filename = "rir_S2-ULA.wav"
        else:
            filename = "rir-S1-ULA.wav"

        wav_path = os.path.join(self.root_dir, "ula", filename)
        pos_path = os.path.join(self.root_dir, "ula", "pos-ULA.csv")

        waveform, sr = torchaudio.load(wav_path)

        if os.path.exists(pos_path):
            df_pos = pd.read_csv(pos_path)
            coords = torch.tensor(df_pos.values, dtype=torch.float32)
        else:
            coords = None

        return {
            "type": "ULA",
            "wav": waveform,       # Shape: [64, Time]
            "sample_rate": sr,
            "coordinates": coords,
            "meta": {"source": source}
        }

    def get_room_geom_plot(self, highlight_mic: typing.Optional[typing.List[typing.Tuple[int, int]]] = None):
        """Plot room geometry showing sources and microphone positions."""
        fig, ax = plt.subplots(figsize=(8, 6))

        # 1. 绘制声源 (Sources)
        sources_path = os.path.join(self.root_dir, "pos-sources.csv")
        if os.path.exists(sources_path):
            src_df = pd.read_csv(sources_path)
            # 假设CSV列是 x, y, z
            ax.scatter(src_df.iloc[:, 0], src_df.iloc[:, 1],
                       c='red', marker='*', s=200, label='Sources')
            for idx, row in src_df.iterrows():
                ax.text(row.iloc[0], row.iloc[1], f" S{idx+1}", fontsize=12)

        # 2. 绘制 HOM 网格
        mic_x, mic_y = [], []

        # 遍历所有 pos 文件读取坐标
        # 为了速度，这里只读取 csv，不加载音频
        for r in self.ROWS:
            for c in self.COLS:
                pos_path = os.path.join(
                    self.root_dir, "hom", f"row{r}", f"pos-R{r}-HOM{c}.csv")
                if os.path.exists(pos_path):
                    df = pd.read_csv(pos_path)
                    mic_x.append(df.iloc[0, 0])
                    mic_y.append(df.iloc[0, 1])

        ax.scatter(mic_x, mic_y, c='blue', marker='o',
                   alpha=0.6, label='HOM Mics')

        # 3. 高亮特定麦克风
        if highlight_mic:
            hx, hy = [], []
            for (r, c) in highlight_mic:
                pos_path = os.path.join(
                    self.root_dir, "hom", f"row{r}", f"pos-R{r}-HOM{c}.csv")
                if os.path.exists(pos_path):
                    df = pd.read_csv(pos_path)
                    hx.append(df.iloc[0, 0])
                    hy.append(df.iloc[0, 1])
            ax.scatter(hx, hy, c='green', marker='x', s=100,
                       linewidths=3, label='Highlight')

        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_title("HOMULA-RIR Room Geometry")
        ax.legend()
        ax.grid(True)
        return fig, ax

    @staticmethod
    def download(
        save_dir,
        logger=print,
        *,
        by_internal_downloader: bool = False,
        headers: typing.Optional[typing.Dict[str, str]] = None,
        tool_preference: typing.Optional[str] = None,
        min_bytes: int = 10 * 1024 * 1024,
    ):
        """Download and extract the dataset (prefer curl/wget; internal on opt-in).

        When external tools are not available, pass ``by_internal_downloader=True``
        to use a requests-based downloader. Extra HTTP headers can be provided
        via ``headers``.
        """
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        zip_path = os.path.join(save_dir, HomulaRIR.ZIP_NAME)

        if os.path.exists(zip_path) and os.path.getsize(zip_path) < 1024 * 1024:
            logger(
                f"Existing file {HomulaRIR.ZIP_NAME} is too small (corrupted), removing it...")
            os.remove(zip_path)

        if not os.path.exists(zip_path):
            logger(f"Downloading from Zenodo to {save_dir}...")
            try:
                HomulaRIR.download_file(
                    HomulaRIR.LINK,
                    zip_path,
                    by_internal_downloader=by_internal_downloader,
                    headers=headers,
                    tool_preference=tool_preference,
                )
            except DownloadError as e:
                raise RuntimeError(str(e))
        else:
            logger("Zip file already exists and size looks okay, skipping download.")

        if os.path.getsize(zip_path) < min_bytes:
            raise RuntimeError(f"Downloaded file too small: {zip_path}")

        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(save_dir)
            logger("Done!")
        except zipfile.BadZipFile:
            logger("Error: The downloaded file is not a valid zip file.")
            logger(f"Please delete {zip_path} and try again.")
            raise


# --- 使用示例 ---
if __name__ == "__main__":
    dataset = HomulaRIR(root_dir="./data/HOMULA-RIR", download=False)

    print(f"Total HOM samples: {len(dataset)}")

    sample = dataset[0]
    print(f"Sample 0 shape: {sample['wav'].shape}, Source: {sample['meta']}")

    # 2. 精确查找 (API 模式)
    hom_data = dataset.get_high_order_microphone(row=3, col=3, source="S1")
    print(f"Center Mic (R3, C3) shape: {hom_data['wav'].shape}")

    # 3. 获取 ULA 数据
    ula_data = dataset.get_uniform_linear_array(source="S2")
    print(f"ULA (S2) shape: {ula_data['wav'].shape}")  # 应该是 [64, Time]

    try:
        dataset.get_room_geom_plot(highlight_mic=[(1, 1), (5, 5)])
    except Exception as e:
        print(f"Plotting skipped: {e}")
