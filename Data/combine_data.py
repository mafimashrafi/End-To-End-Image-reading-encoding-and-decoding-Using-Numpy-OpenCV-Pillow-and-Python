import os
import pandas as pd 
from pathlib import Path

def create_ocr_dataset_with_labels():
    dataset_root = "/Users/mashrafirahman/Documents/E2E_OCR_Project/Data/Dataset"
    data = []
    
    for folder in os.listdir(dataset_root):
        folder_path = os.path.join(dataset_root, folder)
        
        if os.path.isdir(folder_path):
            label = folder
            
            for file in os.listdir(folder_path):
                if file.endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(folder_path, file)
                    
                    data.append({
                        'image_path':img_path,
                        'image_name':file,
                        'label':label,
                        'folder':folder
                    })
                    
    df = pd.DataFrame(data)
    return df

ocr_df = create_ocr_dataset_with_labels()
print(ocr_df.head(10))
print(f"\nTotal images: {len(ocr_df)}")
print(f"\nUnique labels: {ocr_df['label'].nunique()}")
print(f"\nLabel distribution:\n{ocr_df['label'].value_counts()}")

ocr_df.to_csv('dataset_index.csv', index=False)