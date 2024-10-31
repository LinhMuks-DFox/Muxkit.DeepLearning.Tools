import random

import torch

from lib.AudioSet.utils import tensor_masking, create_mask_chunk_2d


class TimeSequenceLengthFixer(torch.nn.Module):
    """
    Fix the length of time sequence.
    >>> import torchaudio
    >>> wav_form, sample_rate = torchaudio.load("test.wav")
    >>> fixer = TimeSequenceLengthFixer(5, sample_rate)
    >>> fixed_wav_form = fixer(wav_form)
    >>> fixed_wav_form.shape
    """
    _FIX_MODE = {
        "random", "r", "random-timezone",
        "s", "start",
        "e", "end"
    }

    def __init__(self, fixed_length: int, sample_rate: int, mode="r"):
        super().__init__()
        if mode not in self._FIX_MODE:
            raise ValueError(f"Invalid mode:{mode}, mode shall be {self._FIX_MODE}")
        self.mode_ = mode
        self.fixed_length = fixed_length * sample_rate

    def forward(self, audio_data: torch.Tensor) -> torch.Tensor:
        if self.mode_ in {"r", "random", "random-timezone"}:
            if audio_data.shape[1] < self.fixed_length:
                return self.select_time_zone(audio_data, 0)[0]
            return self.select_time_zone(audio_data,
                                         random.randint(0, audio_data.shape[1] - self.fixed_length)
                                         )[0]
        if self.mode_ in {"s", "start"}:
            return self.select_time_zone(audio_data, 0)[0]
        if self.mode_ in {"e", "end"}:
            return self.select_time_zone(audio_data, audio_data.shape[1] - self.fixed_length)[0]
        else:
            raise ValueError(f"Invalid mode:{self.mode_}")

    def select_time_zone(self, audio_data: torch.Tensor, start_time: int):
        if audio_data.shape[1] < self.fixed_length:
            audio_data = torch.nn.functional.pad(audio_data,
                                                 (0, self.fixed_length - audio_data.shape[1]))
        return audio_data[:, start_time: start_time + self.fixed_length], start_time


class SoundTrackSelector(torch.nn.Module):
    """
    Select one track from stereo audio.
    >>> import torchaudio
    >>> wav_form, sample_rate = torchaudio.load("test.wav") # where test.wav is a stereo audio
    >>> selector = SoundTrackSelector("left") # see _VALID_TRACKS for valid tracks
    >>> selected_wav_form = selector(wav_form)
    >>> selected_wav_form.shape
    """
    _VALID_TRACKS = {"all", "left", "right", "mix", "random-single", "random"}
    _MOD_SELECT_KERNEL = {
        "all": lambda x: x,
        "left": lambda x: x[0].unsqueeze(0),
        "right": lambda x: x[1].unsqueeze(0),
        "mix": lambda x: x.mean(dim=0).unsqueeze(0),
    }

    def __init__(self, mode: str):
        super().__init__()
        if mode not in self._VALID_TRACKS:
            raise ValueError(f"mode must be one of {', '.join(self._VALID_TRACKS)}, but got {mode}")
        if mode == "random":
            mode = random.choice(["all", "left", "right", "mix"])
        if mode == "random-single":
            mode = random.choice(["left", "right", "mix"])
        self.select_mode = mode
        self.select_kernel = self._MOD_SELECT_KERNEL[mode]

    def forward(self, audio_data: torch.Tensor) -> torch.Tensor:
        return self.select_kernel(audio_data)


class TimeSequenceMaskingTransformer(torch.nn.Module):
    """
    Mask the time sequence of audio data.
    >>> import torchaudio
    >>> wav_form, sample_rate = torchaudio.load("test.wav")
    >>> masker = TimeSequenceMaskingTransformer(0.1)
    >>> masked_wav_form, unmasked_wav_form = masker(wav_form)
    >>> masked_wav_form.shape
    >>> unmasked_wav_form.shape
    """

    def __init__(self, mask_rate: float):
        self.mask_rate_ = mask_rate
        super().__init__()

    def forward(self, audio_data: torch.Tensor):
        masked_audio_data, unmasked_audio_data, mask = tensor_masking(audio_data, self.mask_rate_)
        return masked_audio_data, unmasked_audio_data


class SpectrogramMaskingTransformer(torch.nn.Module):
    """
    Create a masked spectrogram.
    >>> import torchaudio
    >>> wav_form, sample_rate = torchaudio.load("test.wav")
    >>> masker = SpectrogramMaskingTransformer(0.1)
    >>> masked_wav_form, unmasked_wav_form = masker(wav_form)
    >>> masked_wav_form.shape
    >>> unmasked_wav_form.shape
    >>> masker.get_mask().shape
    """

    def __init__(self, mask_rate: float):
        self.mask_rate_ = mask_rate
        super().__init__()
        self.prev_mask_ = None

    def forward(self, audio_data: torch.Tensor):
        mask = create_mask_chunk_2d(audio_data.shape[1:], self.mask_rate_)
        masked_audio_data, unmasked_audio_data, mask = tensor_masking(audio_data, self.mask_rate_, mask=mask)
        self.prev_mask_ = mask
        return masked_audio_data, unmasked_audio_data

    def get_mask(self):
        return self.prev_mask_


if __name__ == "__main__":
    time_fixer = TimeSequenceLengthFixer(5, 16000)
    print(time_fixer(torch.randn([1, 160000])).shape)
