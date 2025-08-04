import pandas as pd

@data_loader
def load_data():
    url = "/mnt/c/Users/anefu/Desktop/AI/mlopszc_proj_orc/spine_mage_pipeline/data/Dataset_spine.csv"
    print(f'Reading the data from {url}')
    df = pd.read_csv(url)
    return df