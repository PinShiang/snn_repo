#!/usr/bin/env python3
"""
Overlap/merge two audio files by adding them together.
Process all audio files in an input folder and save merged files with 'noisy_' prefix.

Usage:
    Basic usage with default noise file:
        python merge_audio.py <input_folder> <output_folder>
    
    Specify custom noise file:
        python merge_audio.py <input_folder> <output_folder> --noise-file path/to/noise.wav
    
    Customize mix ratio (0.0 = only input, 1.0 = only noise, 0.5 = equal mix):
        python merge_audio.py <input_folder> <output_folder> --mix-ratio 0.3
    
    Full example:
        python merge_audio.py ./input ./output --noise-file gaussian_noise_1.4s.wav --mix-ratio 0.5

Examples:
    # Process all audio files in ./input folder, save to ./output with default noise file
    python merge_audio.py ./input ./output
    
    # Use custom noise file and 30% noise mix
    python merge_audio.py ./input ./output --noise-file ./noise/gaussian_noise_1.4s.wav --mix-ratio 0.3
    
    # Process with absolute paths
    python merge_audio.py /path/to/input /path/to/output --noise-file /path/to/noise.wav

Output:
    All processed files will be saved in the output folder with the prefix 'noisy_'.
    For example: input.wav -> noisy_input.wav
"""

import numpy as np
import soundfile as sf
from pathlib import Path
import argparse
import sys


def merge_audio_files(file1_path, file2_path, output_path=None, mix_ratio=0.5):
    """
    Merge two audio files by overlapping/adding them together.
    
    Args:
        file1_path: Path to first audio file
        file2_path: Path to second audio file
        output_path: Output file path (default: file1_path with "_merged" suffix)
        mix_ratio: Ratio for mixing (0.0 = only file1, 1.0 = only file2, 0.5 = equal mix)
    
    Returns:
        Path to the merged output file
    """
    # Load both audio files
    audio1, sr1 = sf.read(file1_path)
    audio2, sr2 = sf.read(file2_path)
    
    print(f"Loaded {Path(file1_path).name}: {len(audio1)} samples, {sr1} Hz")
    print(f"Loaded {Path(file2_path).name}: {len(audio2)} samples, {sr2} Hz")
    
    # Check sample rates
    if sr1 != sr2:
        print(f"Warning: Different sample rates ({sr1} vs {sr2}). Resampling audio2 to {sr1} Hz...")
        try:
            from scipy import signal
            # Resample audio2 to match audio1's sample rate
            num_samples = int(len(audio2) * sr1 / sr2)
            audio2 = signal.resample(audio2, num_samples)
            sr2 = sr1
        except ImportError:
            print("Warning: scipy not available. Using simple resampling...")
            # Simple resampling: repeat or skip samples
            ratio = sr1 / sr2
            indices = np.round(np.arange(len(audio2)) * ratio).astype(int)
            indices = np.clip(indices, 0, len(audio2) - 1)
            audio2 = audio2[indices]
            sr2 = sr1
    
    # Handle stereo/mono conversion
    if audio1.ndim > 1:
        audio1 = audio1[:, 0] if audio1.shape[1] > 0 else audio1.mean(axis=1)
    if audio2.ndim > 1:
        audio2 = audio2[:, 0] if audio2.shape[1] > 0 else audio2.mean(axis=1)
    
    # Make both arrays the same length
    min_len = min(len(audio1), len(audio2))
    max_len = max(len(audio1), len(audio2))
    
    # Trim or pad to match lengths
    if len(audio1) > len(audio2):
        audio1_trimmed = audio1[:min_len]
        audio2_padded = audio2
        # If audio2 is shorter, loop it or pad with zeros
        if len(audio2) < len(audio1):
            # Option 1: Loop the noise
            num_loops = (len(audio1) // len(audio2)) + 1
            audio2_padded = np.tile(audio2, num_loops)[:len(audio1)]
        audio1_final = audio1
        audio2_final = audio2_padded
    elif len(audio2) > len(audio1):
        audio2_trimmed = audio2[:min_len]
        audio1_padded = audio1
        # Loop or pad audio1
        if len(audio1) < len(audio2):
            num_loops = (len(audio2) // len(audio1)) + 1
            audio1_padded = np.tile(audio1, num_loops)[:len(audio2)]
        audio1_final = audio1_padded
        audio2_final = audio2
    else:
        audio1_final = audio1
        audio2_final = audio2
    
    # Mix the audio files
    # mix_ratio: 0.0 = only file1, 1.0 = only file2, 0.5 = equal mix
    mixed = (1 - mix_ratio) * audio1_final + mix_ratio * audio2_final
    
    # Normalize to prevent clipping
    max_val = np.max(np.abs(mixed))
    if max_val > 1.0:
        mixed = mixed / max_val
        print(f"Normalized output (max was {max_val:.3f})")
    
    # Determine output path
    if output_path is None:
        file1_stem = Path(file1_path).stem
        output_path = Path(file1_path).parent / f"{file1_stem}_merged.wav"
    else:
        output_path = Path(output_path)
    
    # Save merged audio
    sf.write(str(output_path), mixed, sr1)
    
    print(f"\n✓ Merged audio saved to: {output_path}")
    print(f"  Duration: {len(mixed) / sr1:.3f} seconds")
    print(f"  Sample rate: {sr1} Hz")
    print(f"  Mix ratio: {mix_ratio} (file1: {1-mix_ratio}, file2: {mix_ratio})")
    
    return output_path


def process_folder(input_folder, output_folder, noise_file_path, mix_ratio=0.5):
    """
    Process all audio files in the input folder and merge them with noise.
    
    Args:
        input_folder: Path to folder containing input audio files
        output_folder: Path to folder where output files will be saved
        noise_file_path: Path to the noise file to merge with
        mix_ratio: Ratio for mixing (0.0 = only input, 1.0 = only noise, 0.5 = equal mix)
    """
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)
    noise_file_path = Path(noise_file_path)
    
    # Validate paths
    if not input_folder.exists():
        print(f"Error: Input folder '{input_folder}' not found!")
        sys.exit(1)
    
    if not noise_file_path.exists():
        print(f"Error: Noise file '{noise_file_path}' not found!")
        sys.exit(1)
    
    # Create output folder if it doesn't exist
    output_folder.mkdir(parents=True, exist_ok=True)
    
    # Supported audio file extensions
    audio_extensions = {'.wav', '.flac', '.mp3', '.ogg', '.m4a', '.aac'}
    
    # Find all audio files in input folder
    audio_files = [f for f in input_folder.iterdir() 
                   if f.is_file() and f.suffix.lower() in audio_extensions]
    
    if not audio_files:
        print(f"Warning: No audio files found in '{input_folder}'")
        return
    
    print(f"Found {len(audio_files)} audio file(s) to process")
    print(f"Noise file: {noise_file_path.name}")
    print(f"Output folder: {output_folder}\n")
    
    # Process each audio file
    for i, audio_file in enumerate(audio_files, 1):
        print(f"[{i}/{len(audio_files)}] Processing: {audio_file.name}")
        
        # Create output filename with 'noisy_' prefix
        output_filename = f"noisy_{audio_file.name}"
        output_path = output_folder / output_filename
        
        try:
            merge_audio_files(
                audio_file,
                noise_file_path,
                output_path=output_path,
                mix_ratio=mix_ratio
            )
        except Exception as e:
            print(f"  ✗ Error processing {audio_file.name}: {e}")
            continue
    
    print(f"\n✓ Success! Processed {len(audio_files)} file(s)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Merge audio files in a folder with noise file, adding 'noisy_' prefix to outputs"
    )
    parser.add_argument(
        "input_folder",
        type=str,
        help="Path to folder containing input audio files"
    )
    parser.add_argument(
        "output_folder",
        type=str,
        help="Path to folder where output files will be saved"
    )
    parser.add_argument(
        "--noise-file",
        type=str,
        default="gaussian_noise_1.4s.wav",
        help="Path to noise file (default: gaussian_noise_1.5s.wav in script directory)"
    )
    parser.add_argument(
        "--mix-ratio",
        type=float,
        default=0.5,
        help="Mix ratio (0.0 = only input, 1.0 = only noise, 0.5 = equal mix, default: 0.5)"
    )
    
    args = parser.parse_args()
    
    # If noise file is relative, check in script directory first
    noise_path = Path(args.noise_file)
    if not noise_path.is_absolute() and not noise_path.exists():
        script_dir = Path(__file__).parent
        noise_path = script_dir / args.noise_file
    
    process_folder(
        args.input_folder,
        args.output_folder,
        noise_path,
        mix_ratio=args.mix_ratio
    )

