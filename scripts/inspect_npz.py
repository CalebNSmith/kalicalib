import argparse
import numpy as np
from pathlib import Path
import sys

def inspect_npz(npz_path):
    """
    Inspect the contents of an NPZ file and print detailed information.
    
    Args:
        npz_path (str): Path to the NPZ file
    """
    try:
        # Load the NPZ file
        data = np.load(npz_path)
        
        # Get basic file info
        file_size = Path(npz_path).stat().st_size
        print(f"\n=== NPZ File Information ===")
        print(f"File path: {npz_path}")
        print(f"File size: {file_size / 1024:.2f} KB")
        
        # List all arrays in the file
        print(f"\n=== Arrays in NPZ file ===")
        print(f"Total number of arrays: {len(data.files)}")
        
        # Count arrays by type
        empty_arrays = 0
        point_arrays = 0
        
        # Detailed information for each array
        for i, name in enumerate(data.files, 1):
            array = data[name]
            print(f"\n{i}. Array: {name}")
            print(f"   Shape: {array.shape}")
            print(f"   Data type: {array.dtype}")
            print(f"   Memory size: {array.nbytes / 1024:.2f} KB")
            
            if array.size == 0:
                empty_arrays += 1
                print("   Status: Empty array")
                continue
                
            # Calculate basic statistics if numeric and non-empty
            if np.issubdtype(array.dtype, np.number):
                try:
                    print(f"   Min value: {array.min()}")
                    print(f"   Max value: {array.max()}")
                    print(f"   Mean value: {array.mean():.4f}")
                    print(f"   Non-zero elements: {np.count_nonzero(array)}")
                    
                    # Count 0s and 1s
                    num_zeros = np.sum(array == 0)
                    num_ones = np.sum(array == 1)
                    print(f"   Number of 0s: {num_zeros}")
                    print(f"   Number of 1s: {num_ones}")
                    
                except Exception as e:
                    print(f"   Error calculating statistics: {str(e)}")
            
            # Show sample of array contents for non-empty arrays
            print(f"   First few elements: {array.flatten()[:5]}")
            
            # Special handling for point coordinate arrays (shape: Nx2)
            if len(array.shape) == 2 and array.shape[1] == 2:
                point_arrays += 1
                print(f"   Number of points: {len(array)}")
                if len(array) > 0:
                    print(f"   Point range - X: [{array[:,1].min():.1f}, {array[:,1].max():.1f}]")
                    print(f"   Point range - Y: [{array[:,0].min():.1f}, {array[:,0].max():.1f}]")

        # Print summary
        print(f"\n=== Summary ===")
        print(f"Total arrays: {len(data.files)}")
        print(f"Empty arrays: {empty_arrays}")
        print(f"Point coordinate arrays: {point_arrays}")
        print(f"Other arrays: {len(data.files) - empty_arrays - point_arrays}")

    except Exception as e:
        print(f"Error inspecting NPZ file: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        if 'data' in locals():
            data.close()

def main():
    parser = argparse.ArgumentParser(description="Inspect the contents of an NPZ file")
    parser.add_argument("npz_file", help="Path to the NPZ file to inspect")
    args = parser.parse_args()
    
    inspect_npz(args.npz_file)

if __name__ == "__main__":
    main()