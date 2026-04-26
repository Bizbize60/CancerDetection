import pandas as pd
di = pd.read_csv(r"C:\Users\PC\Desktop\Dataset\csv\dicom_info.csv")
print(di[["file_path", "image_path"]].head(10).to_string())
print("\nUnique SeriesDescription values:")
print(di["SeriesDescription"].value_counts() if "SeriesDescription" in di.columns else "yok")