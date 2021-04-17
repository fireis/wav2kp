import torchaudio
import pandas as pd

def read_audio_file(path=""):
    """
    Reads the specifies file in path and returns it in Tensor format
    """
    return torchaudio.load(path)

if __name__ == '__main__':
    # TODO: MAKE THIS READ TEH PATH FROM ARGPARSE
    file_path = os.path.abspath("").replace("data", "docs") + "/train_val_test_dist.xlsx"
    train_test_dist = pd.read_excel(file_path)
    train_files = train_test_dist
