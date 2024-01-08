# Fine-Tuning Framework

This repository contains a fine-tuning framework for training and evaluating transformer-based language models using Hugging Face's `transformers` library and `trl`. The main script, `sft.py`, provides a comprehensive pipeline for dataset preparation, model loading, training, evaluation, and logging.

## Features

- **Dataset Preparation**: Supports custom dataset preparation with tokenization and mixing.
- **Model Loading**: Loads pre-trained models with support for quantization and PEFT (Parameter-Efficient Fine-Tuning).
- **Training and Evaluation**: Implements a training loop with support for mixed precision and distributed training.
- **Logging and Monitoring**: Integrates with Weights & Biases (WandB) for experiment tracking and Hugging Face Hub for model sharing.
- **Custom Configurations**: Uses YAML-based configuration files for flexible setup.

## Requirements

- PyTorch
- Hugging Face `transformers`
- `trl` library
- Weights & Biases (`wandb`)

Install the required dependencies using:

```bash
pip install -r requirements.txt
```

## Usage

### 1. Prepare Configuration

Create a YAML configuration file specifying `DataConfig`, `ModelConfig`, and `SFTConfig`. Refer to the `src/configs` module for details on configuration options.

### 2. Run the Script

Execute the script with the configuration file as an argument:

```bash
python sft.py path/to/config.yaml
```

### 3. Key Functionalities

- **Dataset Preparation**: Prepares datasets using `prepare_datasets` and tokenizes them with `get_tokenizer`.
- **Model Loading**: Loads pre-trained models with optional quantization and PEFT configurations.
- **Training**: Trains the model using `SFTTrainer` with support for mixed precision and distributed training.
- **Evaluation**: Evaluates the model and computes metrics like perplexity.
- **Logging**: Logs metrics and examples to WandB and saves the model to Hugging Face Hub if enabled.

### 4. Example Configuration

Below is an example YAML configuration:

```yaml
DataConfig:
    dataset_mixer:
        dataset_name: "your_dataset"
    ...

ModelConfig:
    model_name_or_path: "gpt2"
    torch_dtype: "auto"
    ...

SFTConfig:
    do_train: true
    do_eval: true
    output_dir: "./output"
    ...
```

## Logging and Monitoring

- **Weights & Biases**: Logs training metrics, examples, and summaries. Initialize WandB with `wandb.init()` before training.
- **Hugging Face Hub**: Pushes the trained model and model card to the Hugging Face Hub if `push_to_hub` is enabled.

## Notes

- The script supports mixed precision training (`fp16`/`bf16`) for memory efficiency.
- Quantization is supported for 4-bit and 8-bit models.
- Ensure proper setup of distributed training if running on multiple GPUs.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgments

- Hugging Face `transformers` and `trl` libraries.
- Weights & Biases for experiment tracking.
