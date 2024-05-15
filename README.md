# Fine-Tuning Llama3 using QLoRA with :hugs: Transformers

This project offers minimal tools and scripts for efficiently fine-tuning the Llama3 models, primarily using the :hugs: Transformers library. Datasets suitable for commercial use, such as [kunishou/databricks-dolly-15k-ja](https://huggingface.co/datasets/kunishou/databricks-dolly-15k-ja) and [kunishou/oasst2-135k-ja](https://huggingface.co/datasets/kunishou/oasst2-135k-ja), are provided by default to facilitate practical applications.

For distributed training, we opt for [Accelerate](https://github.com/huggingface/accelerate) and [Deepspeed](https://github.com/microsoft/DeepSpeed) due to their ease of use. Example configurations can be found in the `examples/` folder.

NOTE: This project does not involve using recent alternatives like [unsloth](https://github.com/unslothai/unsloth/tree/main) or [torchtune](https://github.com/pytorch/torchtune) for fine-tuning, but focuses solely on Transformers.

## Getting Started

### Requirements

- Python 3.10 or higher
- CUDA 12.1 or higher
- Compatible NVIDIA GPU (Ampere or higher)

### Install from Source

To install scripts and package from the source, follow the commands below. It is recommended to use `venv` for creating a virtual environment.

```bash
git clone https://github.com/keishihara/finetuning_llama3_hf.git
cd finetuning_llama3_hf
python3 -m venv env
source env/bin/activate
pip install -U pip setuptools wheel packaging
pip install -e ".[full]"
```

You may encounter missing Python dependencies when running the provided scripts. If so, please install them one at a time as they are needed.

### Huggingface Cache Management

If you store :hugs: Transformers models or datasets in a specific directory, it's useful to change the default cache directory from the system default (`~/.cache/huggingface`) to one that suits your setup:

```bash
export HF_HOME=/path/to/your/cache/directory/huggingface
```

Check if the correct cache path is set successfully:

```bash
python -c "from transformers import file_utils; print(file_utils.default_cache_path)"
```

For more details, please visit [this page](https://huggingface.co/docs/transformers/main/en/installation#cache-setup).

## Examples of Fine-Tuning Llama3

### Supervised Fine-Tuning

Example fine-tuning scripts are provided in the [001_sft_llama3](/examples/001_sft_llama3/) directory along with Accelerate configs. To run distributed (single node, 8 processes) instruction-tuning on the dolly-ja and oasst2 datasets using QLoRA, follow the commands below:
```bash
cd examples/001_sft_llama3
./runs/train_llama3.sh
```

If you have a single consumer gpu like a GeForce RTX 4090, a single process script is also provided:
```bash
cd examples/001_sft_llama3
# Around 18GB of VRAM required (8bit quantized)
CUDA_VISIBLE_DEVICES=0 ./runs/train_llama3_on_single_gpu.sh
```

If you want to train on your custom dataset, an example script for using a custom dataset is provided in the
[002_sft_llama3_on_custom_dataset](/examples/002_sft_llama3_on_custom_dataset/) directory:
```bash
cd examples/002_sft_llama3_on_custom_dataset/
./runs/train_llama3_on_custom_dataset.sh
```

## License

The scripts provided in this repository are distributed under the terms of the Apache-2.0 license.

The licensing for the models and datasets follows the respective terms set by their providers.
