import csv
import glob
import os

def main():
    # Find the CSV file in the current directory (ignoring our dataset CSVs)
    csv_files = [f for f in glob.glob("*.csv") if "metadata" not in f and "clinical" not in f and "cognitive" not in f]

    if not csv_files:
        print("ERROR: Could not find the ADNI CSV file. Please move it into the 'CV Lab' folder.")
        return

    # Use the most recently downloaded CSV
    csv_file = sorted(csv_files, key=lambda x: os.path.getmtime(x), reverse=True)[0]
    print(f"Reading from: {csv_file}")

    baseline_ids = []
    with open(csv_file, 'r', encoding='utf-8', errors='replace') as f:
        reader = csv.DictReader(f)
        for row in reader:
            visit = row.get('Visit', '').strip().lower()
            if visit in ['sc', 'bl']:
                img_id = row.get('Image Data ID') or row.get('Image ID')
                if img_id:
                    baseline_ids.append(img_id.strip())

    # Remove duplicates just in case
    baseline_ids = list(set(baseline_ids))
    
    if not baseline_ids:
        print("No baseline scans found in the CSV!")
        return

    output_str = ",".join(baseline_ids)
    print(f"\nFound {len(baseline_ids)} baseline scans!")
    
    with open("baseline_image_ids.txt", "w") as out_f:
        out_f.write(output_str)
        
    print("Success! All IDs have been saved to 'baseline_image_ids.txt'.")
    print("Open that text file, copy everything, and paste it into the 'Image ID' box in ADNI Advanced Search!")

if __name__ == "__main__":
    main()
