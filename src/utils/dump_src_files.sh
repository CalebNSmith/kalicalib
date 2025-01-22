#!/bin/bash

# Output file path
OUTPUT_FILE="src/utils/python_files_dump.txt"

# Clear or create the output file
> "$OUTPUT_FILE"

# Find all Python files recursively in src/, exclude __init__.py files
find src/ -type f -name "*.py" ! -name "__init__.py" | while read -r file; do
    # Add file name as header with markdown formatting
    echo -e "\n# File: $file\n" >> "$OUTPUT_FILE"
    echo "'''python" >> "$OUTPUT_FILE"
    cat "$file" >> "$OUTPUT_FILE"
    echo -e "'''\n" >> "$OUTPUT_FILE"
    echo -e "---\n" >> "$OUTPUT_FILE"
done

find scripts/ -type f -name "train.py" ! -name "__init__.py" | while read -r file; do
    # Add file name as header with markdown formatting
    echo -e "\n# File: $file\n" >> "$OUTPUT_FILE"
    echo "'''python" >> "$OUTPUT_FILE"
    cat "$file" >> "$OUTPUT_FILE"
    echo -e "'''\n" >> "$OUTPUT_FILE"
    echo -e "---\n" >> "$OUTPUT_FILE"
done


find configs/ -type f -name "default.yaml" ! -name "__init__.py" | while read -r file; do
    # Add file name as header with markdown formatting
    echo -e "\n# File: $file\n" >> "$OUTPUT_FILE"
    echo "'''" >> "$OUTPUT_FILE"
    cat "$file" >> "$OUTPUT_FILE"
    echo -e "'''\n" >> "$OUTPUT_FILE"
    echo -e "---\n" >> "$OUTPUT_FILE"
done

find scripts/ -type f -name "inference.py" ! -name "__init__.py" | while read -r file; do
    # Add file name as header with markdown formatting
    echo -e "\n# File: $file\n" >> "$OUTPUT_FILE"
    echo "'''python" >> "$OUTPUT_FILE"
    cat "$file" >> "$OUTPUT_FILE"
    echo -e "'''\n" >> "$OUTPUT_FILE"
    echo -e "---\n" >> "$OUTPUT_FILE"
done


find scripts/ -type f -name "generate_single_heatmap.py" ! -name "__init__.py" | while read -r file; do
    # Add file name as header with markdown formatting
    echo -e "\n# File: $file\n" >> "$OUTPUT_FILE"
    echo "'''python" >> "$OUTPUT_FILE"
    cat "$file" >> "$OUTPUT_FILE"
    echo -e "'''\n" >> "$OUTPUT_FILE"
    echo -e "---\n" >> "$OUTPUT_FILE"
done

echo "Python files have been dumped to $OUTPUT_FILE"