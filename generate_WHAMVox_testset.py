"""
# WHAMVox Test Set

This test set is a mix of speech files from the VoxCeleb2 [1] test set and the
WHAM [2] noise recordings test set.

Download these datasets (for VoxCeleb, you only need to download the test part), unzip
them and specify the unzipped path as arguments to this script.

## Requirements:

The following requirements need to installed for this script to run:
```
audioread==2.1.9
librosa==0.8.0
numpy~=1.19.4
pandas~=1.2.0
scipy~=1.6.0
soundfile
```

## References

[1] https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox2.html
[2] http://wham.whisper.ai/
"""
import logging
import os
import platform
from numbers import Number
from typing import Tuple, Union, List, Optional

import audioread
import librosa.core
import numpy as np
import pandas as pd
import soundfile
from scipy import stats

if platform.system() == "Darwin":
    # Disable core audio because it causes audioread to fail on certain m4a files for
    # some reason...
    audioread._ca_available = lambda: False
    print(
        "Disabled Core Audio. Available audioread backends are:",
        audioread.available_backends(),
    )

VALUE_RANGE = {
    np.dtype("int16"): (-(2 ** 15), 2 ** 15 - 1),
    np.dtype("float32"): (-1, 1),
}
RANDOM_SEED = 42


def sample_snr_values(
    num_samples: int,
    loc: float = 8.0,
    scale: float = 7.0,
    seed: Union[int, None] = RANDOM_SEED,
) -> List[float]:
    """
    Sample SNR values from an SNR distribution. The default values result in a
    distribution that is similar to the one in [1].

    Args:
        num_samples: number of samples to draw from the distribution
        loc: mean of the normal distribution
        scale: variance of the normal distribution
        seed: random seed used for drawing the values

    Returns:
        list of SNR values, length of the list is `num_samples`

    References:
        [1] https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5824438/figure/F4/
    """
    logger = logging.getLogger("WHAMVox_testset.get_snr_values")

    dist = stats.norm(loc=loc, scale=scale)
    values = dist.rvs(random_state=seed, size=(num_samples,))  # type: np.ndarray
    logger.info(
        f"Sampled {num_samples} SNR values from normal distribution with "
        f"loc={loc:.4f} and scale={scale:0.4f} using random seed {seed}."
    )
    return list(values)


def _dBFS_to_absolute(
    value: Union[Number, np.ndarray], range_max: float
) -> Union[Number, np.ndarray]:
    """
    Convert dBFS to absolute amplitude value. The absolute value depends on the data
    range. The formula is `value_absolute = 10 ** (value_dBFS/20) * range_max`.

    Args:
        value: value in dBFS
        range_max: maximum peak value

    Returns:
        value in absolute scale
    """
    return 10 ** (value / 20) * range_max


def pad_and_crop(
    data_speech: np.ndarray,
    data_noise: np.ndarray,
    sample_rate: int,
    duration: float,
    pad_duration: float = 0.1,
):
    """
    Pad speech by `pad_duration` seconds by adding silence at the beginning,
    then crop both speech and noise such that they are `duration` seconds long.

    Cropping is done by removing samples at the end of the waveform.

    Args:
        data_speech: speech data audio waveform as numpy array
        data_noise: noise data audio waveform as numpy array
        sample_rate: audio sample rate in Hz
        duration: duration of the cropped audio in seconds
        pad_duration: duration in seconds of silence to pad at the beginning of the
            speech audio

    Returns:
        padded and cropped speech, cropped noise
    Raises:
        ValueError: if either padded speech or noise is shorter than `duration`
    """
    # First, pad speech by adding zeros at the beginning.
    if pad_duration > 0:
        pad_frames = int(pad_duration * sample_rate)
        pad_shape = (pad_frames,) + data_speech.shape[1:]
        zeros = np.zeros(shape=pad_shape, dtype=data_speech.dtype)
        data_speech = np.concatenate((zeros, data_speech), axis=0)

    # Now check that both noise and speech have the right duration.
    min_frames = int(duration * sample_rate)
    if data_speech.shape[0] < min_frames or data_noise.shape[0] < min_frames:
        raise ValueError(
            f"Speech and/or noise data is shorter than minimum "
            f"duration of {min_frames:d} frames."
        )
    # And finally crop the speech and noise.
    data_speech = data_speech[:min_frames, ...]
    data_noise = data_noise[:min_frames, ...]
    return data_speech, data_noise


def mix(
    data_speech: np.ndarray,
    data_noise: np.ndarray,
    snr: float,
    rms: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Mix speech and noise audio data together with the given SNR value.

    The mixing formulate is:
    ```
    data_mixed = r1* data_speech + r2 * data_noise.
    ```

    We have that
    ```
    factor = rms_speech / rms_noise
    SNR = 20log10(r1/r2 * factor)
    r1/r2 = 10 ** (SNR/20) / factor.
    ```
    where `rms_speech` and `rms_noise` are the RMS values of speech and noise.

    Let `x = 10 ** (SNR/20) / factor` and assuming that `r1 + r2 = 1`, this leads to
    ```
    r2 = 1 / (x + 1) and r1 = 1 - r2
    ```
    The mixed audio is then rescaled so that it's RMS is the desired value.

    The mixed audio is clipped according to the value range corresponding to the audio
    data type.

    This function also returns the rescaled speech and noise such that
    `audio_mixed = audio_speech + audio_noise`.

    Examples:
        >>> import librosa.core
        >>> speech, _ = librosa.core.load("some_speech_file.wav")
        >>> noise, _ = librosa.core.load("some_noise_file.wav")
        >>> mixed, speech, noise = mix(
        ...     data_speech=speech,
        ...     data_noise=noise,
        ...     snr=5,  # dB
        ...     rms=-20,  # dBFS,
        ... )

    Args:
        data_speech: speech data numpy array
        data_noise: noise data numpy array
        snr: desired SNR value in dB
        rms: desired RMS value dB FS

    Returns:
        mixed audio data, scaled speech data and scaled noise data
    """
    # Remove DC offset in speech and noise.
    data_speech = data_speech - np.mean(data_speech)
    data_noise = data_noise - np.mean(data_noise)

    if data_speech.dtype != data_noise.dtype:
        raise TypeError(
            f"Speech and noise data do not have the same data "
            f"type: {data_speech.dtype} != {data_noise.dtype}."
        )
    dtype = data_speech.dtype

    if dtype not in VALUE_RANGE:
        raise TypeError(
            f"Data type '{dtype}' is not supported. The audio should have on the "
            f"following data types: {', '.join(t.name for t in VALUE_RANGE)}."
        )
    # Get theoretical min and max value for codec
    range_min, range_max = VALUE_RANGE[dtype]

    # We convert the given RMS value from dBFS to absolute.
    rms_absolute = _dBFS_to_absolute(rms, range_max)

    rms_speech = np.sqrt(np.mean(np.square(data_speech)))
    rms_noise = np.sqrt(np.mean(np.square(data_noise)))

    factor = rms_speech / rms_noise

    # Do the remaining computations in float64 for better precision and avoiding
    # overflows (if original dtype is an integer dtype).
    data_speech = data_speech.astype(np.float64)
    data_noise = data_noise.astype(np.float64)

    # Mix the two signals. We first need to calculate the mixing proportions
    # r1 and r2 required to achieve the desired SNR value.
    x = 10 ** (snr / 20) / factor
    r2 = 1 / (x + 1)
    r1 = 1 - r2

    data_mixed = r1 * data_speech + r2 * data_noise
    data_speech = r1 * data_speech
    data_noise = r2 * data_noise

    # Now scale the mixed signal so that it has the desired RMS value.
    scale_factor = rms_absolute / np.std(data_mixed)
    data_mixed = data_mixed * scale_factor
    data_speech = data_speech * scale_factor
    data_noise = data_noise * scale_factor

    # Finally, clip the values.
    data_mixed = data_mixed.clip(range_min, range_max)
    data_speech = data_speech.clip(range_min, range_max)
    data_noise = data_noise.clip(range_min, range_max)

    # Convert data back to original data type.
    data_mixed = data_mixed.astype(dtype)
    data_speech = data_speech.astype(dtype)
    data_noise = data_noise.astype(dtype)

    return data_mixed, data_speech, data_noise


def main(
    file_csv: str,
    path_wham: str,
    path_vox: str,
    destination: str,
    sample_rate: int = 22050,
    rms: float = -20.0,
    duration: float = 4.0,
    snr_distribution_loc: Optional[float] = None,
    snr_distribution_scale: Optional[float] = None,
):
    logger = logging.getLogger("WHAMVox_testset")

    logger.info(f"Loading CSV file '{file_csv}'.")
    frame = pd.read_csv(file_csv)
    logger.info(f"CSV file contains {len(frame):d} entries.")

    if snr_distribution_scale is not None and snr_distribution_loc is not None:
        logger.info(
            f"Sampling new SNR values to replace values provided in the CSV file."
        )
        frame["SNR"] = sample_snr_values(
            num_samples=len(frame),
            loc=snr_distribution_loc,
            scale=snr_distribution_scale,
        )
    else:
        logger.info(f"Using SNR values provided in CSV file.")

    os.makedirs(destination, exist_ok=True)

    for i, row in frame.iterrows():
        file_speech = os.path.join(path_vox, row["file_speech"])
        file_noise = os.path.join(path_wham, row["file_noise"])

        data_speech, _ = librosa.core.load(file_speech, sr=sample_rate)
        data_noise, _ = librosa.core.load(file_noise, sr=sample_rate)

        data_speech, data_noise = pad_and_crop(
            data_speech,
            data_noise,
            sample_rate,
            duration,
        )

        data_mixed, data_label, _ = mix(
            data_speech=data_speech, data_noise=data_noise, snr=row["SNR"], rms=rms
        )
        file_mixed = os.path.join(destination, f"{i:04d}.wav")
        file_label = os.path.join(destination, f"{i:04d}_label.wav")
        logger.debug(
            f"Mixed speech file '{row['file_speech']}' with noise file "
            f"'{row['file_noise']}' at SNR '{row['SNR']}'."
        )
        logger.debug(
            f"Saving mixed file under filename '{os.path.basename(file_mixed)}'"
        )
        soundfile.write(file=file_mixed, data=data_mixed, samplerate=sample_rate)
        soundfile.write(file=file_label, data=data_label, samplerate=sample_rate)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--file_csv",
        type=str,
        required=True,
        help="Path to the CSV file containing the list of speech files, "
        "noise files and SNR values.",
    )
    parser.add_argument(
        "--path_wham",
        type=str,
        required=True,
        help="Path to the unzipped WHAM dataset. This folder should contain "
        "the 'tt' subfolder where the testing noise file should be.",
    )
    parser.add_argument(
        "--path_vox",
        type=str,
        required=True,
        help="Path to the unzipped VoxCeleb2 dataset. This folder should contain "
        "the 'aac' subfolder where all the audio files are, grouped by speaker ID.",
    )
    parser.add_argument(
        "--dest",
        type=str,
        required=True,
        help="Destination folder where the generated test set audio "
        "files will be saved. The folder will be created if it doens't already exist.",
    )
    parser.add_argument(
        "--sample_rate",
        type=int,
        required=False,
        default=22050,
        help="Sample rate in Hz of the generated audio files. The noise and "
        "speech files will be resample to have this sample rate.",
    )
    parser.add_argument(
        "--rms",
        type=float,
        required=False,
        default=-20.0,
        help="RMS value in dB Full Scale which controls the loudness of the generated "
        "audio files. All generated audio waveforms will have this root mean "
        "square value.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        required=False,
        default=4.0,
        help="Duration in seconds of the generated samples. The default duration "
        "is 4 seconds. Longer durations will result in less samples being created "
        "since only samples where both speech and noise and longer than the "
        "minimum duration will be generated.",
    )
    parser.add_argument(
        "--snr_distribution_loc",
        type=float,
        required=False,
        default=None,
        help="If specified, draw SNR values from a normal distribution with this mean "
        "instead of taking the values from the CSV file. Both "
        "'--snr_distribution_loc' and '--snr_distribution_scale' must be "
        "specified in order to use a custom SNR distribution.",
    )
    parser.add_argument(
        "--snr_distribution_scale",
        type=float,
        required=False,
        default=None,
        help="If specified, draw SNR values from a normal distribution with this "
        "variance instead of taking the values from the CSV file. Both "
        "'--snr_distribution_loc' and '--snr_distribution_scale' must be "
        "specified in order to use a custom SNR distribution.",
    )
    args = parser.parse_args()
    main(
        file_csv=args.file_csv,
        path_vox=args.path_vox,
        path_wham=args.path_wham,
        destination=args.dest,
        rms=args.rms,
        duration=args.duration,
        sample_rate=args.sample_rate,
        snr_distribution_loc=args.snr_distribution_loc,
        snr_distribution_scale=args.snr_distribution_scale,
    )
