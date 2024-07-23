import pandas as pd
import glob

folder_path = './data/raw/reviews'
file_paths = glob.glob(f'{folder_path}/*.csv')
save_path = f'./data/processed/merged_reviews.csv'

all_data = []
success_count = 0
fail_count = 0
for file in file_paths:
    try:
        df = pd.read_csv(file)
        all_data.append(df)
        success_count += 1
    except Exception as e:
        print(f"Error reading file {file}: {e}")
        fail_count += 1

if all_data:
    all_data = pd.concat(all_data, ignore_index=True)
    
    all_data.to_csv(save_path, index=False)
    print(f'Merged file saved as {save_path}')
else:
    print("No files were successfully read.")

print(f"Number of files successfully: {success_count}")
print(f"Number of files failed: {fail_count}")