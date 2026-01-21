"""
JVS (Japanese Versatile Speech) dataset utilities and Dataset wrapper.

Reference: JVS corpus – free Japanese multi-speaker voice corpus.
https://arxiv.org/abs/1908.06248
"""

import os
import pathlib
import typing
import torch
import torchaudio
from torch.utils.data import Dataset, Subset

class JVSDataset(Dataset):
    """
    A PyTorch Dataset for the JVS (Japanese Versatile Speech) Corpus.

    This class implements a "Lazy Loading" strategy combined with a Hash Map Index
    to ensure low memory usage and O(1) query performance.

    Paper:
        "JVS corpus: free Japanese multi-speaker voice corpus"
        https://arxiv.org/abs/1908.06248

    Attributes:
        STYLES (List[str]): Supported speaking styles.
            - 'parallel100': 100 parallel sentences (shared across all speakers).
            - 'nonpara30': 30 non-parallel sentences (unique to each speaker).
            - 'whisper10': 10 whispered sentences (subset of parallel100).
            - 'falset10': 10 falsetto sentences (subset of parallel100).
        GDRIVE_ID (str): Google Drive ID for downloading the dataset.
    """

    STYLES = ['parallel100', 'nonpara30', 'whisper10', 'falset10']
    GDRIVE_ID = "19oAw8wWn3Y7z6CKChRdAyGOB9yupL_Xt"

    def __init__(self, 
                 root_dir: typing.Union[str, pathlib.Path], 
                 sample_rate: int = 24000, 
                 download: bool = False):
        """
        Initializes the JVS Dataset.

        Args:
            root_dir (str | Path): The root directory containing the 'jvs_ver1' folder.
            sample_rate (int, optional): The target sample rate for audio loading. 
                Audio will be resampled on-the-fly if it does not match. Defaults to 24000.
            download (bool, optional): If True, downloads the dataset from Google Drive 
                before initialization. Defaults to False.

        Raises:
            FileNotFoundError: If the dataset is not found and download is False.
        """
        self.root_dir = pathlib.Path(root_dir)
        self.target_sr = sample_rate
        
        if download:
            self.download(self.root_dir)

        # Flat registry for __getitem__ (O(1) access by integer index)
        self.samples: typing.List[dict] = []

        # Hierarchical Hash Map for query (O(1) access by logical keys)
        # Structure: { spk_id: { style: { sent_id: global_index } } }
        self.index_map: typing.Dict[int, typing.Dict[str, typing.Dict[int, int]]] = {}

        self._build_index()
    def _build_index(self):
            """
            Scans the directory structure to build the internal index and sample list.
            """
            if not self.root_dir.exists():
                raise FileNotFoundError(f"Root directory {self.root_dir} not found.")

            # Filter and sort speaker directories
            spk_dirs = sorted([d for d in self.root_dir.iterdir() if d.is_dir() and d.name.startswith('jvs')])
            
            for spk_dir in spk_dirs:
                try:
                    spk_id = int(spk_dir.name.replace("jvs", "")) # e.g., jvs001 -> 1
                except ValueError:
                    continue

                self.index_map[spk_id] = {}

                for style in self.STYLES:
                    style_dir = spk_dir / style
                    if not style_dir.exists():
                        continue

                    self.index_map[spk_id][style] = {}
                    
                    # 1. Load Transcripts
                    trans_file = style_dir / "transcripts_utf8.txt"
                    text_map = {}
                    if trans_file.exists():
                        with open(trans_file, 'r', encoding='utf-8') as f:
                            for line in f:
                                if ':' in line:
                                    fid, txt = line.strip().split(':', 1)
                                    text_map[fid] = txt

                    # 2. Scan Audio Files (Changed to rglob for recursive search)
                    # JVS audio is often in a subdir like 'wav24kHz'
                    wav_files = sorted(list(style_dir.rglob("*.wav")))
                    
                    for wav_path in wav_files:
                        fid = wav_path.stem
                        
                        # Parse Sentence ID
                        try:
                            sent_id = int(fid.split('_')[-1])
                        except ValueError:
                            sent_id = -1

                        # Create Metadata
                        meta = {
                            "path": str(wav_path),
                            "text": text_map.get(fid, ""),
                            "spk_id": spk_id,
                            "style": style,
                            "sent_id": sent_id,
                            "filename": fid
                        }
                        
                        # Update Indices
                        current_idx = len(self.samples)
                        self.samples.append(meta)
                        self.index_map[spk_id][style][sent_id] = current_idx


    def __len__(self) -> int:
        """Returns the total number of samples in the dataset."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> typing.Dict[str, typing.Any]:
        """
        Retrieves a sample by its global index.

        Args:
            idx (int): The global index of the sample.

        Returns:
            dict: A dictionary containing the sample data:
                - 'audio' (Tensor): The audio waveform, shape [1, T].
                - 'text' (str): The transcript text.
                - 'spk_id' (int): The speaker ID (1-100).
                - 'style' (str): The speaking style (e.g., 'parallel100').
                - 'sent_id' (int): The sentence ID (useful for alignment).
                - 'sample_rate' (int): The sampling rate of the audio.
        """
        meta = self.samples[idx]
        
        # Lazy Loading
        wav, sr = torchaudio.load(meta['path'])
        
        # Resampling
        if sr != self.target_sr:
            wav = torchaudio.functional.resample(wav, sr, self.target_sr)
            
        return {
            "audio": wav,
            "text": meta['text'],
            "spk_id": meta['spk_id'],
            "style": meta['style'],
            "sent_id": meta['sent_id'],
            "sample_rate": self.target_sr,
            "path": meta['path']
        }

    # ==========================================
    # Core Engine: Universal Query
    # ==========================================

    def query(self, 
              spk_ids: typing.Optional[typing.List[int]] = None, 
              styles: typing.Optional[typing.List[str]] = None,
              sent_ids: typing.Optional[typing.List[int]] = None) -> typing.List[int]:
        """
        Queries the dataset for samples matching the specified criteria.

        This method uses the internal Hash Map Index to perform O(1) lookups per key,
        avoiding full dataset iteration.

        Args:
            spk_ids (List[int], optional): A list of speaker IDs to include. 
                If None, includes all speakers.
            styles (List[str], optional): A list of styles to include. 
                Must be subset of self.STYLES. If None, includes all styles.
            sent_ids (List[int], optional): A list of sentence IDs to include. 
                If None, includes all sentences.

        Returns:
            List[int]: A sorted list of global indices corresponding to the matching samples.
        """
        found_indices = []
        
        # If None, use all available keys from the index
        target_spks = spk_ids if spk_ids is not None else self.index_map.keys()
        
        for spk in target_spks:
            if spk not in self.index_map: continue
            
            target_styles = styles if styles is not None else self.index_map[spk].keys()
            
            for style in target_styles:
                if style not in self.index_map[spk]: continue
                
                style_map = self.index_map[spk][style]
                
                if sent_ids is None:
                    # Select all sentences for this style
                    found_indices.extend(style_map.values())
                else:
                    # Select specific sentences
                    for sid in sent_ids:
                        if sid in style_map:
                            found_indices.append(style_map[sid])
                            
        return sorted(found_indices)

    # ==========================================
    # High-Level APIs (Powered by Query)
    # ==========================================

    def get_sample(self, spk_id: int, sent_id: int, style: str = 'nonpara30') -> typing.Dict:
        """
        Retrieves a single sample using logical identifiers.

        Args:
            spk_id (int): The speaker ID.
            sent_id (int): The sentence ID.
            style (str, optional): The speaking style. Defaults to 'nonpara30'.

        Returns:
            dict: The sample dictionary (see __getitem__).

        Raises:
            ValueError: If the specified sample does not exist in the index.
        """
        indices = self.query(spk_ids=[spk_id], styles=[style], sent_ids=[sent_id])
        if not indices:
            raise ValueError(f"Sample not found: Spk {spk_id}, Style {style}, Sent {sent_id}")
        return self[indices[0]]

    def get_parallel(self, 
                     spk_ids: typing.List[int], 
                     style: str = 'parallel100') -> typing.Tuple[typing.List[torch.Tensor], ...]:
        """
        Retrieves aligned parallel audio data for a list of speakers.

        This method automatically calculates the intersection of available sentence IDs
        across the requested speakers and returns the audio tensors in alignment.

        Args:
            spk_ids (List[int]): The list of speaker IDs to retrieve.
            style (str, optional): The style of parallel data to retrieve.
                Must be a style where sentence IDs are consistent across speakers
                (e.g., 'parallel100', 'whisper10', 'falset10'). 
                Defaults to 'parallel100'.
                
                Note: Using 'nonpara30' here will likely result in an empty tuple,
                as sentence IDs are not designed to be aligned in non-parallel data.

        Returns:
            Tuple[List[Tensor]]: A tuple of lists, where each list contains audio tensors 
            for one speaker. 
            Format: ( [Spk1_SentA, Spk1_SentB], [Spk2_SentA, Spk2_SentB], ... )
        """
        if not spk_ids:
            return ()

        # 1. Determine common sentence IDs (Intersection)
        # Check first speaker
        if spk_ids[0] not in self.index_map or style not in self.index_map[spk_ids[0]]:
            return tuple([] for _ in spk_ids)
            
        common_sents = set(self.index_map[spk_ids[0]][style].keys())
        
        # Intersect with remaining speakers
        for spk in spk_ids[1:]:
            if spk in self.index_map and style in self.index_map[spk]:
                common_sents &= set(self.index_map[spk][style].keys())
            else:
                common_sents = set()
                break
        
        sorted_sents = sorted(list(common_sents))
        if not sorted_sents:
            return tuple([] for _ in spk_ids)

        # 2. Batch Retrieval
        batch_out = []
        for spk in spk_ids:
            # We use direct dictionary lookup here for speed, bypassing query() overhead
            indices = [self.index_map[spk][style][sid] for sid in sorted_sents]
            
            # Load actual audio tensors
            wavs = [self[i]['audio'] for i in indices]
            batch_out.append(wavs)
            
        return tuple(batch_out)

    def subset(self, 
               spk_ids: typing.Optional[typing.List[int]] = None, 
               styles: typing.Optional[typing.List[str]] = None) -> Subset:
        """
        Creates a torch.utils.data.Subset containing only the filtered samples.

        Args:
            spk_ids (List[int], optional): Allowed speaker IDs.
            styles (List[str], optional): Allowed styles.

        Returns:
            torch.utils.data.Subset: A subset dataset compatible with DataLoader.
        """
        indices = self.query(spk_ids=spk_ids, styles=styles)
        return Subset(self, indices)

    @staticmethod
    def download_with_gdown(root_dir: typing.Union[str, pathlib.Path], logger: typing.Callable = print):
        """Download JVS via gdown and extract.

        Raises ImportError if gdown is not available.
        """
        try:
            import gdown  # type: ignore
            import zipfile
        except ImportError as e:
            raise ImportError("JVSDataset download requires 'gdown'. Install via: pip install gdown") from e

        root = pathlib.Path(root_dir)
        root.mkdir(parents=True, exist_ok=True)

        if (root / "jvs001").exists():
            logger("JVS dataset appears to exist. Skipping download.")
            return

        url = f'https://drive.google.com/uc?id={JVSDataset.GDRIVE_ID}'
        output = root / "jvs_ver1.zip"

        logger(f"Downloading JVS dataset to {output} via gdown...")
        gdown.download(url, str(output), quiet=False)

        logger("Extracting archive...")
        with zipfile.ZipFile(output, 'r') as zip_ref:
            zip_ref.extractall(root)
        logger("Download and extraction complete.")

    @staticmethod
    def download(root_dir: typing.Union[str, pathlib.Path], logger: typing.Callable = print):
        """Wrapper that requires gdown. See ``download_with_gdown``."""
        JVSDataset.download_with_gdown(root_dir, logger)


if __name__ == "__main__":
    # Example Usage Script
    from torch.utils.data import DataLoader

    # 1. Setup
    print(">>> Initializing JVSDataset...")
    # Ensure download=True for the first run
    dataset = JVSDataset(root_dir="./data/JVS-org/jvs_ver1/", download=False)
    print(f"Dataset initialized. Total valid samples: {len(dataset)}")

    # 2. Standard DataLoader Usage (e.g., for TTS training)
    print("\n[A] Testing Standard DataLoader")
    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    for batch in loader:
        print(f"  Random Batch: Spk{batch['spk_id'].item()} | Style: {batch['style'][0]}")
        print(f"  Audio Shape: {batch['audio'].shape}")
        break 

    # 3. Parallel Data Usage (e.g., for Voice Conversion)
    print("\n[B] Testing Parallel Data Alignment")
    spks = [1, 2]
    # Explicitly requesting 'parallel100' style
    # You could also request 'whisper10' or 'falset10' here
    aligned_data = dataset.get_parallel(spk_ids=spks, style='parallel100')
    
    if aligned_data and len(aligned_data[0]) > 0:
        spk1_wavs, spk2_wavs = aligned_data
        print(f"  Found {len(spk1_wavs)} aligned pairs for Spk {spks}.")
        print(f"  Spk{spks[0]} Sample 1 Shape: {spk1_wavs[0].shape}")
        print(f"  Spk{spks[1]} Sample 1 Shape: {spk2_wavs[0].shape}")
    else:
        print("  No aligned data found.")

    # 4. Specific Query Usage (e.g., Debugging)
    print("\n[C] Testing Specific Sample Query")
    try:
        # Looking for Speaker 10, Sentence 1, Whisper style
        sample = dataset.get_sample(spk_id=10, sent_id=1, style='whisper10')
        print(f"  Retrieved: {sample['path']}")
        print(f"  Text: {sample['text']}")
    except ValueError as e:
        print(f"  Query failed: {e}")

    # 5. Subset Usage (e.g., Splitting Train/Val)
    print("\n[D] Testing Subset Creation")
    # Create a subset of only 'nonpara30' data
    nonpara_subset = dataset.subset(styles=['nonpara30'])
    print(f"  Created subset with 'nonpara30' only: {len(nonpara_subset)} samples.")
