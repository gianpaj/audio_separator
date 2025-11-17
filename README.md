# Audio🔹Separator

A powerful audio separation tool that extracts vocals and instrumental tracks from audio files using advanced MDX-Net models. Built with both a Gradio web interface and Replicate API support.

---

## Features

🎵 **Dual Stem Separation**
- Extract vocals or instrumental tracks from any audio file
- High-quality separation powered by MDX-Net models

🎚️ **Audio Effects Processing**
- Apply professional vocal effects including reverb, compression, and EQ
- Customizable effect parameters for instrumental tracks
- Independent effect chains for vocals and background

💻 **Multiple Interfaces**
- Web-based Gradio interface for easy local use
- Replicate API integration for programmatic access
- Command-line support via Cog

🔧 **Format Support**
- Input: MP3, WAV, FLAC, and other common formats
- Output: WAV or MP3
- Automatic stereo conversion and normalization

⚡ **Performance**
- GPU acceleration with CUDA 12.1 support
- CPU fallback for systems without GPU
- Optimized model inference with ONNX Runtime

---

## Quick Start

### Prerequisites

- Python 3.11+
- FFmpeg
- CUDA 12.1 (optional, for GPU acceleration)
- PyTorch 2.5.1+

### Installation

1. **Clone the repository**
   ```bash
   git clone https://huggingface.co/spaces/r3gm/Audio_separator
   cd Audio_separator
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Usage

#### Web Interface (Gradio)

```bash
python app.py
```

Then open your browser to `http://localhost:7860`

#### Replicate API (Cog)

**Local testing:**
```bash
cog predict -i audio=@your_audio.mp3 -i extract_vocals=true -i output_format=wav
```

**Build and deploy:**
```bash
cog build
cog push r8.im/your-username/audio-separator
```

---

## API Reference

### Cog Predict Endpoint

**Inputs:**

- `audio` (Path): Input audio file
- `extract_vocals` (bool): 
  - `true` (default): Extract and process vocal track
  - `false`: Extract instrumental track
- `output_format` (str): 
  - `wav` (default): Uncompressed WAV format
  - `mp3`: Compressed MP3 format

**Output:**
- Returns the separated audio file in the requested format

**Example:**
```bash
cog predict \
  -i audio=@song.mp3 \
  -i extract_vocals=true \
  -i output_format=wav
```

---

## Audio Effects

### Vocal Effects (Applied when extract_vocals=true)

- **Reverb**: Room-like ambience (room_size: 0.15, damping: 0.7)
- **Compressor**: Dynamic range control (threshold: -15dB, ratio: 4.0)
- **Gain**: Volume normalization (0dB)
- **Highpass Filter**: Remove unwanted low frequencies

### Instrumental Effects (Applied when extract_vocals=false)

- **Highpass Filter**: Remove very low frequencies
- **Lowpass Filter**: Clean up high frequencies
- **Reverb**: Add space and depth
- **Compressor**: Smooth dynamic response
- **Gain**: Volume adjustment

---

## Architecture

### Core Components

**MDX Model** (`predict.py` / `app.py`)
- ONNX-based neural network for stem separation
- Operates on 44.1kHz stereo audio
- Processes audio in chunks for memory efficiency

**Audio Processing Pipeline**
1. Load and normalize input audio
2. Convert to stereo WAV if needed
3. Run MDX separation model
4. Apply vocal or instrumental effects
5. Convert to requested output format

**Supported Models**
- `UVR-MDX-NET-Voc_FT.onnx`: Vocal separation model (primary)
- Additional models automatically downloaded from GitHub releases

### Dependencies

- **PyTorch**: Deep learning framework
- **ONNX Runtime**: Model inference with GPU support
- **Librosa**: Audio analysis and I/O
- **SoundFile**: WAV file handling
- **Pedalboard**: Audio effects processing
- **Gradio**: Web interface
- **FFmpeg**: Format conversion

---

## Configuration

### Application Settings

All default parameters are configurable through Gradio sliders:

**Vocal Effects:**
- Reverb room size: 0.15
- Reverb damping: 0.7
- Reverb wet level: 0.2
- Compressor threshold: -15dB
- Compressor ratio: 4.0
- Compressor attack: 1.0ms
- Compressor release: 100ms
- Gain: 0dB

**Instrumental Effects:**
- Highpass filter: 80Hz
- Lowpass filter: 18000Hz
- Reverb room size: 0.3
- Reverb damping: 0.6
- Compressor threshold: -20dB
- Compressor ratio: 3.0

---

## Files

- **app.py**: Gradio web interface with full effect controls
- **predict.py**: Cog-compatible prediction endpoint for Replicate
- **utils.py**: Utility functions for file handling and logging
- **cog.yaml**: Cog configuration for containerized deployment
- **requirements.txt**: Python package dependencies
- **packages.txt**: System package dependencies
- **pre-requirements.txt**: Pre-installation requirements

---

## Performance

### GPU Requirements
- NVIDIA GPU with CUDA 12.1 support
- Minimum 4GB VRAM recommended
- Tested on A40, RTX 3090, RTX 4090

### Processing Times (Approximate)
- 3-minute song: 15-30 seconds (GPU)
- 3-minute song: 2-5 minutes (CPU)

---

## Troubleshooting

**FFmpeg not found**
```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt-get install ffmpeg

# Windows
choco install ffmpeg
```

**CUDA out of memory**
- Reduce audio length or use CPU processing
- Close other GPU applications

**Poor separation quality**
- Ensure input audio is clear and centered
- Try with different audio sources
- Model works best with 44.1kHz stereo audio

---

## Original Source

Based on the Hugging Face Space: [r3gm/Audio_separator](https://huggingface.co/spaces/r3gm/Audio_separator)

Original repository: https://huggingface.co/spaces/r3gm/Audio_separator/tree/main

This project was adapted into a Replicate Cog using Claude with the following requirements:
- Simplified inputs (audio, extract_vocals, output_format)
- Replicate-compatible prediction interface
- Same defaults as the original app (reverb_room_size: 0.15, reverb_damping: 0.7)

---

## License

MIT License - See LICENSE file for details

---

## Citation

If you use this project in your research or work, please cite:

```bibtex
@misc{audio_separator,
  title={Audio🔹Separator},
  author={r3gm and contributors},
  year={2024},
  howpublish={\url{https://huggingface.co/spaces/r3gm/Audio_separator}}
}
```

---

## Support

For issues, questions, or contributions:
1. Check existing issues on the GitHub repository
2. Create a new issue with detailed description
3. Include sample audio and exact error messages
4. Specify your system configuration (GPU, OS, Python version)

---

## Acknowledgments

- MDX-Net model architecture and weights
- Pedalboard for audio effects
- Librosa for audio processing
- Gradio for the web interface
- Replicate for Cog framework
