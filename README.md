# IMPACT

IMPACT is an adversarial patch generation framework combining Differential Evolution (DE) and a simple Evolution Strategy (ES). It is designed for black-box attacks on image classification models.

## Important Files

- `main.py`: Main entrypoint. Handles argument parsing, model loading, DE + ES optimization loop, logging, and image saving.
- `DE.py`: Differential Evolution related code (population initialization, mutation, crossover, fitness evaluation, decoding individuals, etc.).
- `utils.py`: Data loaders, image processing helpers, logging utilities (`my_logger`) and metric utilities (`my_meter`).

## Environment & Dependencies

Please install the dependencies listed in `requirements.txt` inside an isolated virtual environment. The repository was tested with the following versions:

- matplotlib==3.7.2
- numpy==1.23.5
- pillow==10.2.0
- scikit_learn==1.3.2
- scipy==1.10.1
- seaborn==0.13.2
- timm==1.0.9
- torch==1.12.1
- torchvision==0.13.1

Install example:

```bash
python -m pip install -r requirements.txt
```

## Quick Start (example)

Run the main script from the repository directory. Adjust dataset and checkpoint paths as needed:

```bash
python main.py \
    --dataset ImageNet \
    --data_dir /path/to/imagenet \
    --network ResNet50 \
    --batch_size 1 \
    --dataset_size 100 \
    --population_size 25 \
    --DE_attack_iters 139 \
    --ES_attack_iters 1500 \
    --device cuda:0
```