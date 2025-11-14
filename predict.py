import os
import gc
import hashlib
import json
import shlex
import sys
import subprocess
import tempfile
from typing import List
import librosa
import numpy as np
import soundfile as sf
import torch
from cog import BasePredictor, Input, Path
from tqdm import tqdm
import onnxruntime as ort
import warnings
from pedalboard import Pedalboard, Reverb, Compressor, Gain, HighpassFilter

from utils import (
    create_directories,
    download_manager,
    logger,
)

warnings.filterwarnings("ignore")


class MDXModel:
    def __init__(
        self,
        device,
        dim_f,
        dim_t,
        n_fft,
        hop=1024,
        stem_name=None,
        compensation=1.000,
    ):
        self.dim_f = dim_f
        self.dim_t = dim_t
        self.dim_c = 4
        self.n_fft = n_fft
        self.hop = hop
        self.stem_name = stem_name
        self.compensation = compensation

        self.n_bins = self.n_fft // 2 + 1
        self.chunk_size = hop * (self.dim_t - 1)
        self.window = torch.hann_window(
            window_length=self.n_fft, periodic=True
        ).to(device)

        out_c = self.dim_c

        self.freq_pad = torch.zeros(
            [1, out_c, self.n_bins - self.dim_f, self.dim_t]
        ).to(device)

    def stft(self, x):
        x = x.reshape([-1, self.chunk_size])
        x = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop,
            window=self.window,
            center=True,
            return_complex=True,
        )
        x = torch.view_as_real(x)
        x = x.permute([0, 3, 1, 2])
        x = x.reshape([-1, 2, 2, self.n_bins, self.dim_t]).reshape(
            [-1, 4, self.n_bins, self.dim_t]
        )
        return x[:, :, : self.dim_f]

    def istft(self, x, freq_pad=None):
        freq_pad = (
            self.freq_pad.repeat([x.shape[0], 1, 1, 1])
            if freq_pad is None
            else freq_pad
        )
        x = torch.cat([x, freq_pad], -2)
        x = x.reshape([-1, 2, 2, self.n_bins, self.dim_t]).reshape(
            [-1, 2, self.n_bins, self.dim_t]
        )
        x = x.permute([0, 2, 3, 1])
        x = x.contiguous()
        x = torch.view_as_complex(x)
        x = torch.istft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop,
            window=self.window,
            center=True,
        )
        return x.reshape([-1, 2, self.chunk_size])


class MDX:
    DEFAULT_SR = 44100
    DEFAULT_CHUNK_SIZE = 0 * DEFAULT_SR
    DEFAULT_MARGIN_SIZE = 1 * DEFAULT_SR

    def __init__(
        self, model_path: str, params: MDXModel, processor=0
    ):
        self.device = (
            torch.device(f"cuda:{processor}")
            if processor >= 0
            else torch.device("cpu")
        )
        self.provider = (
            ["CUDAExecutionProvider"]
            if processor >= 0
            else ["CPUExecutionProvider"]
        )

        self.model = params

        self.ort = ort.InferenceSession(model_path, providers=self.provider)
        self.ort.run(
            None,
            {"input": torch.rand(1, 4, params.dim_f, params.dim_t).numpy()},
        )
        self.process = lambda spec: self.ort.run(
            None, {"input": spec.cpu().numpy()}
        )[0]

        self.prog = None

    @staticmethod
    def get_hash(model_path):
        try:
            with open(model_path, "rb") as f:
                f.seek(-10000 * 1024, 2)
                model_hash = hashlib.md5(f.read()).hexdigest()
        except:
            model_hash = hashlib.md5(open(model_path, "rb").read()).hexdigest()

        return model_hash

    @staticmethod
    def segment(
        wave,
        combine=True,
        chunk_size=DEFAULT_CHUNK_SIZE,
        margin_size=DEFAULT_MARGIN_SIZE,
    ):
        if combine:
            processed_wave = None
            for segment_count, segment in enumerate(wave):
                start = 0 if segment_count == 0 else margin_size
                end = None if segment_count == len(wave) - 1 else -margin_size
                if margin_size == 0:
                    end = None
                if processed_wave is None:
                    processed_wave = segment[:, start:end]
                else:
                    processed_wave = np.concatenate(
                        (processed_wave, segment[:, start:end]), axis=-1
                    )

        else:
            processed_wave = []
            sample_count = wave.shape[-1]

            if chunk_size <= 0 or chunk_size > sample_count:
                chunk_size = sample_count

            if margin_size > chunk_size:
                margin_size = chunk_size

            for segment_count, skip in enumerate(
                range(0, sample_count, chunk_size)
            ):
                margin = 0 if segment_count == 0 else margin_size
                end = min(skip + chunk_size + margin_size, sample_count)
                start = skip - margin

                cut = wave[:, start:end].copy()
                processed_wave.append(cut)

                if end == sample_count:
                    break

        return processed_wave

    def pad_wave(self, wave):
        n_sample = wave.shape[1]
        trim = self.model.n_fft // 2
        gen_size = self.model.chunk_size - 2 * trim
        pad = gen_size - n_sample % gen_size

        wave_p = np.concatenate(
            (
                np.zeros((2, trim)),
                wave,
                np.zeros((2, pad)),
                np.zeros((2, trim)),
            ),
            1,
        )

        mix_waves = []
        for i in range(0, n_sample + pad, gen_size):
            waves = np.array(wave_p[:, i:i + self.model.chunk_size])
            mix_waves.append(waves)

        mix_waves = torch.tensor(mix_waves, dtype=torch.float32).to(
            self.device
        )

        return mix_waves, pad, trim

    def _process_wave(self, mix_waves, trim, pad):
        mix_waves = mix_waves.split(1)
        with torch.no_grad():
            pw = []
            for mix_wave in mix_waves:
                if self.prog:
                    self.prog.update()
                spec = self.model.stft(mix_wave)
                processed_spec = torch.tensor(self.process(spec))
                processed_wav = self.model.istft(
                    processed_spec.to(self.device)
                )
                processed_wav = (
                    processed_wav[:, :, trim:-trim]
                    .transpose(0, 1)
                    .reshape(2, -1)
                    .cpu()
                    .numpy()
                )
                pw.append(processed_wav)
        processed_signal = np.concatenate(pw, axis=-1)[:, :-pad]
        return processed_signal

    def process_wave(self, wave: np.array):
        self.prog = tqdm(total=0)
        mix_waves, pad, trim = self.pad_wave(wave)
        self.prog.total = len(mix_waves)
        processed_signal = self._process_wave(mix_waves, trim, pad)
        self.prog.close()
        return processed_signal


def run_mdx(
    model_params,
    output_dir,
    model_path,
    filename,
    suffix=None,
    invert_suffix=None,
    denoise=False,
    device_base="cuda",
):
    if device_base == "cuda":
        device = torch.device("cuda:0")
        processor_num = 0
    else:
        device = torch.device("cpu")
        processor_num = -1

    model_hash = MDX.get_hash(model_path)
    mp = model_params.get(model_hash)
    model = MDXModel(
        device,
        dim_f=mp["mdx_dim_f_set"],
        dim_t=2 ** mp["mdx_dim_t_set"],
        n_fft=mp["mdx_n_fft_scale_set"],
        stem_name=mp["primary_stem"],
        compensation=mp["compensate"],
    )

    mdx_sess = MDX(model_path, model, processor=processor_num)
    wave, sr = librosa.load(filename, mono=False, sr=44100)

    # Normalize input
    peak = max(np.max(wave), abs(np.min(wave)))
    wave /= peak

    if denoise:
        wave_processed = -(mdx_sess.process_wave(-wave)) + (
            mdx_sess.process_wave(wave)
        )
        wave_processed *= 0.5
    else:
        wave_processed = mdx_sess.process_wave(wave)

    # Return to previous peak
    wave_processed *= peak
    stem_name = model.stem_name if suffix is None else suffix

    main_filepath = os.path.join(
        output_dir,
        f"{os.path.basename(os.path.splitext(filename)[0])}_{stem_name}.wav",
    )
    sf.write(main_filepath, wave_processed.T, sr)

    del mdx_sess, wave_processed, wave
    gc.collect()
    torch.cuda.empty_cache()

    return main_filepath


def convert_to_stereo_and_wav(audio_path, output_dir):
    wave, sr = librosa.load(audio_path, mono=False, sr=44100)

    if type(wave[0]) != np.ndarray or audio_path[-4:].lower() != ".wav":
        stereo_path = f"{os.path.splitext(os.path.basename(audio_path))[0]}_stereo.wav"
        stereo_path = os.path.join(output_dir, stereo_path)

        command = shlex.split(
            f'ffmpeg -y -loglevel error -i "{audio_path}" -ac 2 -f wav "{stereo_path}"'
        )
        sub_params = {
            "stdout": subprocess.PIPE,
            "stderr": subprocess.PIPE,
            "creationflags": subprocess.CREATE_NO_WINDOW
            if sys.platform == "win32"
            else 0,
        }
        process_wav = subprocess.Popen(command, **sub_params)
        output, errors = process_wav.communicate()
        if process_wav.returncode != 0 or not os.path.exists(stereo_path):
            raise Exception("Error processing audio to stereo wav")

        return stereo_path
    else:
        return audio_path


def add_vocal_effects(
    input_file,
    output_file,
    reverb_room_size=0.15,
    reverb_damping=0.7,
    reverb_wet_level=0.2,
    compressor_threshold_db=-15,
    compressor_ratio=4.0,
    compressor_attack_ms=1.0,
    compressor_release_ms=100,
    gain_db=0,
):
    effects = [HighpassFilter()]

    effects.append(
        Reverb(
            room_size=reverb_room_size,
            damping=reverb_damping,
            wet_level=reverb_wet_level,
            dry_level=0.8
        )
    )

    effects.append(
        Compressor(
            threshold_db=compressor_threshold_db,
            ratio=compressor_ratio,
            attack_ms=compressor_attack_ms,
            release_ms=compressor_release_ms
        )
    )

    if gain_db:
        effects.append(Gain(gain_db=gain_db))

    board = Pedalboard(effects)

    with sf.SoundFile(input_file) as f:
        audio = f.read(always_2d=True).T
        samplerate = f.samplerate

    effected = board(audio, samplerate)
    sf.write(output_file, effected.T, samplerate)


MDX_DOWNLOAD_LINK = "https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/"
UVR_MODELS = ["UVR-MDX-NET-Voc_FT.onnx"]


class Predictor(BasePredictor):
    def setup(self):
        """Load the models into memory"""
        self.mdxnet_models_dir = "mdx_models"
        os.makedirs(self.mdxnet_models_dir, exist_ok=True)

        # Download required model
        for id_model in UVR_MODELS:
            download_manager(
                os.path.join(MDX_DOWNLOAD_LINK, id_model),
                self.mdxnet_models_dir
            )

        # Load model parameters
        with open(os.path.join(self.mdxnet_models_dir, "data.json")) as infile:
            self.mdx_model_params = json.load(infile)

        # Set device
        self.device_base = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Device: {self.device_base}")
        logger.info(f"ONNX Runtime device: {ort.get_device()}")

    def predict(
        self,
        audio: Path = Input(description="Input audio file URL or path"),
        extract_vocals: bool = Input(
            description="Extract vocals (if False, extracts instrumental)",
            default=True
        ),
        output_format: str = Input(
            description="Output audio format",
            choices=["wav", "mp3"],
            default="wav"
        ),
    ) -> Path:
        """Run audio separation"""

        # Create temp directory for processing
        temp_dir = tempfile.mkdtemp()
        output_dir = os.path.join(temp_dir, "output")
        create_directories(output_dir)

        try:
            # Convert to stereo WAV if needed
            audio_path = str(audio)
            logger.info(f"Processing audio: {audio_path}")
            stereo_path = convert_to_stereo_and_wav(audio_path, temp_dir)

            # Vocal separation
            logger.info("Starting vocal separation...")
            vocals_path = run_mdx(
                self.mdx_model_params,
                output_dir,
                os.path.join(self.mdxnet_models_dir, "UVR-MDX-NET-Voc_FT.onnx"),
                stereo_path,
                suffix="Vocals" if extract_vocals else "Instrumental",
                denoise=True,
                device_base=self.device_base,
            )

            # Apply effects with defaults from app.py
            if extract_vocals:
                logger.info("Applying vocal effects...")
                effects_path = vocals_path.replace(".wav", "_effects.wav")
                add_vocal_effects(
                    vocals_path,
                    effects_path,
                    reverb_room_size=0.15,
                    reverb_damping=0.7,
                    reverb_wet_level=0.2,
                    compressor_threshold_db=-15,
                    compressor_ratio=4.0,
                    compressor_attack_ms=1.0,
                    compressor_release_ms=100,
                    gain_db=0,
                )
                output_path = effects_path
            else:
                output_path = vocals_path

            # Convert format if needed
            if output_format == "mp3":
                wav_root, _ = os.path.splitext(output_path)
                mp3_path = f"{wav_root}.mp3"
                ffmpeg_cmd = [
                    "ffmpeg",
                    "-y",
                    "-i",
                    output_path,
                    "-codec:a",
                    "libmp3lame",
                    "-b:a",
                    "192k",
                    mp3_path,
                ]

                try:
                    subprocess.run(
                        ffmpeg_cmd,
                        check=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                    )
                except FileNotFoundError as exc:
                    raise RuntimeError(
                        "FFmpeg is required for MP3 export but was not found in the runtime."
                    ) from exc
                except subprocess.CalledProcessError as exc:
                    raise RuntimeError(
                        "FFmpeg failed while converting audio to MP3: "
                        f"{exc.stderr.decode(errors='ignore')}"
                    ) from exc

                output_path = mp3_path

            # Move to final output location
            final_output = f"/tmp/output.{output_format}"
            os.rename(output_path, final_output)

            logger.info(f"Processing complete: {final_output}")
            return Path(final_output)

        finally:
            # Cleanup temp directory
            import shutil
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
