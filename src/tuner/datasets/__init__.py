from tuner.datasets.dolly_ja_dataset import get_dolly_ja_dataset
from tuner.datasets.dolly_ja_dataset_sequence import get_dolly_ja_dataset_sequence
from tuner.datasets.oasst2_dataset_sequence import get_oasst2_dataset_sequence
from tuner.datasets.oasst2_ja_dataset_sequence import get_oasst2_ja_dataset_sequence

DATASET_PRESETS = {
    'dolly_ja_dataset': get_dolly_ja_dataset,
    'dolly_ja_dataset_sequence': get_dolly_ja_dataset_sequence,
    'oasst2_dataset_sequence': get_oasst2_dataset_sequence,
    'oasst2_ja_dataset_sequence': get_oasst2_ja_dataset_sequence,
}
