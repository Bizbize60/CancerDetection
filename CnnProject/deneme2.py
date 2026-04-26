import pandas as pd
m = pd.read_csv(r"C:\Users\PC\Desktop\Dataset\csv\mass_case_description_train_set.csv")
print(m.columns.tolist())
print(m[["image file path", "cropped image file path", "ROI mask file path"]].head(5).to_string())