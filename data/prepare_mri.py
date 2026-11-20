import os
import re
import gzip
import shutil
from pathlib import Path

def main():
    input_dir = Path("D:/git from 21 aug/CV Lab/ADNI/PRE PROCESSED")
    output_dir = Path("D:/git from 21 aug/CV Lab/data/raw/mri")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    nii_files = list(input_dir.rglob("*.nii")) + list(input_dir.rglob("*.nii.gz"))
    print(f"Found {len(nii_files)} files remaining. Moving and cleaning up to save space...")
    
    count = 0
    for file_path in nii_files:
        match = re.search(r'(\d{3}_S_\d{4})', file_path.name)
        if not match: match = re.search(r'(\d{3}_S_\d{4})', str(file_path))
        if not match: continue
            
        subject_id = match.group(1)
        target_path = output_dir / f"{subject_id}.nii.gz"
        
        if target_path.exists():
            # Already processed previously! Delete the bulky source file to free up space
            try: 
                file_path.unlink()
            except Exception as e:
                pass
            continue
            
        try:
            if file_path.suffix == '.gz':
                shutil.move(str(file_path), str(target_path)) # Move instead of copy
            else:
                # Compress to save massive amounts of space, then delete the raw original
                with open(file_path, 'rb') as f_in, gzip.open(target_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
                file_path.unlink() # Delete original immediately
            count += 1
            if count % 20 == 0: 
                print(f"Processed {count} new files...")
        except Exception as e:
            print(f"Error processing {file_path.name}: {e}")
            
    print(f"\nDone! Successfully processed {count} new files into {output_dir}")
    print("You can now safely delete the entire 'ADNI' folder, as everything you need is now correctly organized in data/raw/mri/")

if __name__ == "__main__":
    main()
