import os
import glob
import gzip
import nibabel as nib

def check_file(f):
    # 1. Verify complete gzip decompression without EOFError
    with gzip.open(f, 'rb') as gz:
        while True:
            chunk = gz.read(1024 * 1024)
            if not chunk:
                break
    # 2. Verify nibabel can read header
    img = nib.load(f)
    if img.shape is None:
        raise ValueError("Empty or invalid NIfTI volume header")

def main():
    mri_dir = "data/raw/mri"
    files = glob.glob(os.path.join(mri_dir, "*.nii.gz"))
    bad_files = []
    print(f"Thoroughly verifying gzip stream & header for {len(files)} MRI files...")
    
    for i, f in enumerate(files):
        try:
            check_file(f)
        except Exception as e:
            print(f"[{i+1}/{len(files)}] ❌ Corrupted: {os.path.basename(f)} ({e})")
            bad_files.append(f)
    
    for f in bad_files:
        try:
            os.remove(f)
            print(f"  🗑️ Deleted corrupted scan: {os.path.basename(f)}")
        except Exception as e:
            print(f"  ⚠️ Could not delete {f}: {e}")
            
    if len(bad_files) > 0:
        print(f"\nDone. Cleaned {len(bad_files)} truncated/corrupted file(s).")
        print("Dataset loaders will automatically bypass deleted scans or use valid ones.")
    else:
        print("\nAll MRI files are 100% intact and verified.")

if __name__ == "__main__":
    main()
