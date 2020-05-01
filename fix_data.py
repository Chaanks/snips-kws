from utils import load_json, write_json ,load_2_col
from pathlib import Path
import shutil 

def fix_duration(path, duration):
    train = load_json(path)

    for utt in train:
        audio_path = utt['audio_file_path']
        utt['duration'] = float(duration[audio_path])

    save = Path(path.parent, path.stem + '_old' + path.suffix)
    shutil.copyfile(path, save)
    write_json(path, train)


if __name__ == '__main__':
    duration = load_2_col(Path('duration_all_audio_files'))

    #fix_duration(Path('splits/train.json'), duration)
    #fix_duration(Path('splits/dev.json'), duration)
    #fix_duration(Path('splits/test.json'), duration)