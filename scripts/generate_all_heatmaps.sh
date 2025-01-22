#!/bin/bash

# Base directory containing all game folders
DATA_DIR="../data"
# Python script path
SCRIPT_PATH="scripts/generate_single_heatmap.py"

# Create output directories if they don't exist
mkdir -p "../data/kalicalib_v3/images"
mkdir -p "../data/kalicalib_v3/labels"

# Loop through all directories in the data folder
for game_dir in "$DATA_DIR"/*/ ; do
    # Skip the kalicalib directory
    if [[ "$game_dir" == *"kalicalib"* ]]; then
        continue
    fi
    
    # Get the game name from the directory path
    game_name=$(basename "$game_dir")
    
    # Process all jpg files in the images directory
    for image_file in "$game_dir/images"/*.jpg; do
        # Skip if no files found
        [[ -e "$image_file" ]] || continue
        
        # Get the base filename without extension
        base_name=$(basename "$image_file" .jpg)
        
        # Construct the corresponding label path
        label_file="$game_dir/labels/${base_name}.json"
        
        # Skip if label file doesn't exist
        if [[ ! -f "$label_file" ]]; then
            echo "Warning: No label file found for $image_file"
            continue
        fi
        
        # Construct output paths
        image_output="../data/kalicalib_v3/images/${game_name}-${base_name}.jpg"
        label_output="../data/kalicalib_v3/labels/${game_name}-${base_name}.npz"
        
        echo "Processing: $base_name"
        
        # Run the Python script
        python "$SCRIPT_PATH" \
            --image "$image_file" \
            --label "$label_file" \
            --label-output "$label_output" \
            --image-output "$image_output"
            
        if [ $? -ne 0 ]; then
            echo "Error processing $base_name"
        fi
    done
done

echo "Processing complete!"