#!/bin/bash

# Convert audio file(s) to .npz spike file(s)
# Usage: ./convert.sh <input_path> [output_folder]
#   - If input_path is a file: converts that file
#   - If input_path is a folder: converts all audio files in that folder
#   - Supported formats: .wav, .m4a, .mp3, .flac, .ogg
# Example: 
#   ./convert.sh myData/audio/jerry_tao.wav
#   ./convert.sh myData/myOneToTen_audio myData/spikes/myOneToTen

if [ $# -lt 1 ] || [ $# -gt 2 ]; then
    echo "Usage: ./convert.sh <input_path> [output_folder]"
    echo "  input_path: audio file or folder containing audio files"
    echo "  output_folder: (optional) output folder for spike files"
    echo ""
    echo "Examples:"
    echo "  ./convert.sh myData/audio/jerry_tao.wav"
    echo "  ./convert.sh myData/myOneToTen_audio myData/spikes/myOneToTen"
    exit 1
fi

input_path="$1"
output_folder="${2:-}"

# Determine if input is a file or directory
if [ -f "$input_path" ]; then
    # Single file conversion
    input_file="$input_path"
    
    # Get base name without extension
    base_name=$(basename "$input_file")
    base_name="${base_name%.*}"
    
    # Determine output path
    if [ -z "$output_folder" ]; then
        # Default: use myData/spikes/ if input is in myData/audio/
        if [[ "$input_file" == *"myData/audio/"* ]]; then
            output_file="myData/spikes/${base_name}.npz"
        else
            # Use same directory as input
            input_dir=$(dirname "$input_file")
            output_file="${input_dir}/${base_name}.npz"
        fi
    else
        mkdir -p "$output_folder"
        output_file="${output_folder}/${base_name}.npz"
    fi
    
    # Create output directory if it doesn't exist
    mkdir -p "$(dirname "$output_file")"
    
    # Run the conversion
    echo "Converting $input_file to $output_file..."
    python -m lauscher "$input_file" "$output_file" --verbose
    
    if [ $? -eq 0 ]; then
        echo "Conversion successful! Output saved to $output_file"
    else
        echo "Conversion failed for $input_file!"
        exit 1
    fi
    
elif [ -d "$input_path" ]; then
    # Folder conversion
    input_dir="$input_path"
    
    # Determine output directory
    if [ -z "$output_folder" ]; then
        # Default: use myData/spikes/ with same folder name
        if [[ "$input_dir" == *"myData/"* ]]; then
            folder_name=$(basename "$input_dir")
            output_dir="myData/spikes/${folder_name}"
        else
            output_dir="${input_dir}_spikes"
        fi
    else
        output_dir="$output_folder"
    fi
    
    # Create output directory
    mkdir -p "$output_dir"
    
    # Find all audio files (support multiple formats)
    audio_files=$(find "$input_dir" -type f \( -iname "*.wav" -o -iname "*.m4a" -o -iname "*.mp3" -o -iname "*.flac" -o -iname "*.ogg" \) | sort)
    
    if [ -z "$audio_files" ]; then
        echo "Error: No audio files found in '$input_dir'"
        exit 1
    fi
    
    file_count=$(echo "$audio_files" | wc -l)
    echo "Found $file_count audio file(s) in $input_dir"
    echo "Output directory: $output_dir"
    echo ""
    
    success_count=0
    fail_count=0
    
    # Process each file
    while IFS= read -r input_file; do
        if [ -n "$input_file" ]; then
            base_name=$(basename "$input_file")
            base_name="${base_name%.*}"
            output_file="${output_dir}/${base_name}.npz"
            
            echo "[$((success_count + fail_count + 1))/$file_count] Converting $(basename "$input_file")..."
            python -m lauscher "$input_file" "$output_file" --verbose
            
            if [ $? -eq 0 ]; then
                echo "  ✓ Success: $output_file"
                ((success_count++))
            else
                echo "  ✗ Failed: $input_file"
                ((fail_count++))
            fi
            echo ""
        fi
    done <<< "$audio_files"
    
    echo "========================================="
    echo "Conversion complete!"
    echo "  Successful: $success_count"
    echo "  Failed: $fail_count"
    echo "  Output directory: $output_dir"
    
    if [ $fail_count -gt 0 ]; then
        exit 1
    fi
    
else
    echo "Error: '$input_path' is neither a file nor a directory!"
    exit 1
fi

