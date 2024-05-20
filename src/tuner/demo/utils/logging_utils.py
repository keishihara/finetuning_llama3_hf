import time
from pathlib import Path


def setup_logging_dir_and_output_file(model_name_or_path: str, out_dir: str = 'auto') -> str:
    timestamp = time.strftime('%y%m%dT%H%M%S%z')

    if out_dir == 'auto':
        if Path(model_name_or_path).is_dir(): # pretrained model checkpoint
            out_dir = Path(model_name_or_path) / 'predictions'
            local_or_hf = 'local_model'
        else: # hf model id
            out_dir = Path().cwd() / 'predictions' / model_name_or_path.replace('/', '--')
            local_or_hf = model_name_or_path.replace('/', '--')
    else:
        out_dir = Path(out_dir)

    out_file = f'predictions_{local_or_hf}_{timestamp}.jsonl'
    out_dir.mkdir(exist_ok=True)
    return out_dir / out_file
