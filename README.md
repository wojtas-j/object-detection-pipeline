# Object Detection Pipeline

**Object Detection Pipeline** is a flexible and extensible framework for training and evaluating object detection models on various datasets, designed to streamline research and development in computer vision. This project simplifies the process of experimenting with datasets and models, enabling easy integration of new datasets and architectures.

## Features

- **Dataset Management**: Automated downloading, extraction, conversion, and preprocessing of datasets, with support for COCO and BDD100k.
- **Model Training**: Train object detection models (e.g., YOLO) with support for custom modifications (e.g., depthwise-separable layers), fine-tuning, and transfer learning.
- **Model Evaluation**: Evaluate models on images and videos, with comprehensive metrics logging.
- **Custom Metrics**: Enhanced training and evaluation with additional metrics, such as `detection_flicker_rate` and `iou_consistency` for video-based evaluation on BDD100k.
- **Configuration-Driven**: Manage dataset processing, training, and evaluation via intuitive YAML configuration files using Hydra and OmegaConf.
- **Logging**: Robust logging system that outputs to console and saves logs to files (with rotation) in non-test mode.
- **Testing**: Comprehensive unit tests for all major components, ensuring reliability and maintainability.
- **Extensibility**: Abstract architecture allows seamless addition of new datasets and models, facilitating research and experimentation.

## Technologies

- **Python**: Core programming language
- **PyTorch, Torch, Torchvision**: Deep learning frameworks
- **Ultralytics YOLO**: Object detection model
- **Faster R-CNN**: Object detection model
- **OpenCV**: Image and video processing
- **Hydra, OmegaConf**: Configuration management
- **Pytest**: Unit testing framework
- **Logging**: Custom logging with file rotation
- **TensorBoard**: Visualization of training metrics
- **Pycocotools**: COCO dataset utilities
- **CUDA**: GPU acceleration for training and evaluation
- **PyCharm**: Recommended IDE for development

## Requirements

- **Software**:
  ```bash
  pip install -r requirements.txt
  ```
- **Hardware** (tested configuration):
  - CPU: AMD Ryzen 7 7700X
  - GPU: NVIDIA RTX 4060 (8GB VRAM)
  - RAM: 32GB DDR5
  - *Note*: Lower specifications may work but have not been tested.

## Installation

1. Clone the repository:
   ```bash
   git clone git@github.com:wojtas-j/object-detection-pipeline.git
   cd object-detection-pipeline
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Prepare datasets using `dataset_manager` (see Usage).
4. Train or evaluate models using `train_eval_manager` (see Usage).

## Usage

### Dataset Management
Configure dataset downloading, extraction, conversion, and preprocessing in `configs/datasets/dataset_manager.yaml`:

```yaml
manager:
  dataset_dir: datasets
  stage: 1  # 0: download and extract, 1: convert and preprocess
  types:
    - coco
    - bdd100k
```

The `dataset_downloader` and `dataset_converter` modules are designed abstractly, allowing seamless support for multiple datasets and custom processing logic via YAML configurations.

1. **Stage 0**: Download and extract datasets (e.g., COCO, BDD100k). Update paths in `configs/datasets/coco.yaml` or `configs/datasets/bdd100k.yaml` to match extracted data.
   ```bash
   python -m src.datasets.dataset_manager
   ```
2. **Stage 1**: Convert and preprocess datasets (e.g., select classes, generate YOLO-compatible annotations).
   ```bash
   python -m src.datasets.dataset_manager
   ```

### Training and Evaluation
Configure training/evaluation in `configs/training_and_evaluation/train_eval_manager.yaml`:

```yaml
manager:
  mode: 1  # 0: training, 1: evaluation, 2: training and evaluation
  selected_model: "yolo"
  selected_dataset_train: "coco"
  selected_dataset_eval: "coco"
  models:
    yolo:
      name: "yolo"
      datasets_train: ["coco", "bdd100k"]
      datasets_eval: ["coco", "bdd100k"]
```

The `model_trainer` and `model_evaluator` modules are designed abstractly, enabling easy extension by implementing custom model classes (e.g., `YOLOTrainer`, `YOLOEvaluator`) for new architectures.

1. Update dataset-specific configurations (e.g., `configs/training_and_evaluation/train_yolo_coco.yaml` for training, `configs/training_and_evaluation/eval_yolo_coco.yaml` for evaluation) with paths and hyperparameters.
2. Run training and/or evaluation:
   ```bash
   python -m src.training.train_eval_manager
   ```

## Project Structure

```
object-detection-pipeline/
├── .gitignore                              # Git ignore file
├── requirements.txt                        # Project dependencies
├── README.md                               # Project documentation
├── pytest.ini                              # Pytest configuration
├── configs/                                # Configuration files
│   ├── datasets/                           # Dataset configurations
│   │   ├── bdd100k.yaml                    # BDD100k dataset configuration
│   │   ├── coco.yaml                       # COCO dataset configuration
│   │   ├── dataset_manager.yaml            # Dataset manager configuration
│   ├── training_and_evaluation/            # Training and evaluation configurations
│   │   ├── eval_yolo_bdd100k.yaml          # YOLO evaluation for BDD100k
│   │   ├── eval_yolo_coco.yaml             # YOLO evaluation for COCO
│   │   ├── train_yolo_bdd100k.yaml         # YOLO training for BDD100k
│   │   ├── train_yolo_coco.yaml            # YOLO training for COCO
│   │   ├── train_eval_manager.yaml         # Training/evaluation manager configuration
│   │   ├── paths_files/                    # Dataset path configurations
│   │   │   ├── coco_yolo.yaml              # COCO paths and classes for YOLO
│   │   │   ├── bdd100k_yolo.yaml           # BDD100k paths and classes for YOLO
├── datasets/                               # Datasets directory
│   ├── bdd100k/                            # BDD100k dataset
│   │   ├── base_labels/                    # JSON annotations (train/val)
│   │   ├── faster-rcnn/                    # Converted annotations for Faster R-CNN
│   │   ├── images/                         # Images (train/val)
│   │   ├── labels/                         # YOLO annotations (train/val, per-frame .txt files)
│   │   ├── videos/val/                     # Validation videos generated from frames
│   ├── coco/                               # COCO dataset
│   │   ├── annotations/                    # JSON annotations, including Faster R-CNN
│   │   ├── images/                         # Images (train/val)
│   │   ├── labels/                         # YOLO annotations (train/val, per-image .txt files)
├── src/                                    # Source code
│   ├── datasets/                           # Dataset management
│   │   ├── dataset_converter.py            # Dataset annotation conversion
│   │   ├── dataset_downloader.py           # Dataset downloading/extraction
│   │   ├── dataset_manager.py              # Dataset management script
│   │   ├── dataset_utils.py                # Dataset utility functions
│   ├── exceptions/                         # Custom exceptions
│   │   ├── exceptions.py
│   ├── log_config/                         # Logging utilities
│   │   ├── logging_config.py               # Logger configuration
│   │   ├── train_eval_metric_utils.py      # Metrics processing and logging
│   ├── training/                           # Training and evaluation
│   │   ├── model_evaluator.py              # Model evaluation (YOLOEvaluator)
│   │   ├── model_trainer.py                # Model training (YOLOTrainer)
│   │   ├── train_eval_manager.py           # Training/evaluation management
├── tests/                                  # Unit tests
│   ├── datasets/                           # Dataset-related tests
│   │   ├── test_dataset_converter.py
│   │   ├── test_dataset_downloader.py
│   │   ├── test_dataset_manager.py
│   │   ├── test_dataset_utils.py
│   ├── log_config/                         # Logging-related tests
│   │   ├── test_logging_config.py
│   │   ├── test_train_eval_metric_utils.py
│   ├── training/                           # Training/evaluation tests
│   │   ├── test_model_evaluator.py
│   │   ├── test_model_trainer.py
│   │   ├── test_train_eval_manager.py
├── logs/                                   # Logs (generated in non-test mode)
```

## Testing

Run unit tests for all components (except `exceptions.py`) using:
```bash
  pytest tests/ -v
```

## Contributing

Contributions are welcome! You can extend the pipeline by adding new datasets or models, following the guidelines below.

### Adding New Datasets
The `dataset_downloader` and `dataset_converter` modules are designed with abstract base classes (`DatasetDownloader` and `DatasetConverter`), allowing easy extension for new datasets. To add a new dataset:

1. **Implement `DatasetDownloader`**:
   - Create a new class inheriting from `DatasetDownloader`.
   - Implement abstract methods: `get_config_path`, `download`, and `extract`.
   - Example for a new dataset (e.g., `CustomDatasetDownloader`):
     ```python
     from src.datasets.dataset_downloader import DatasetDownloader
     from omegaconf import DictConfig
     from pathlib import Path

     class CustomDatasetDownloader(DatasetDownloader):
         def get_config_path(self) -> str:
             return "custom_dataset"
         
         def download(self, dataset_dir: str | Path, cfg: DictConfig) -> list[Path]:
             return self._download_files(dataset_dir, cfg, "custom_dataset")
         
         def extract(self, downloaded_files: list[Path], cfg: DictConfig, dataset_dir: str | Path) -> None:
             self._extract_files(dataset_dir, cfg, downloaded_files, "custom_dataset")
     ```

2. **Implement `DatasetConverter`**:
   - Create a new class inheriting from `DatasetConverter`.
   - Implement abstract methods: `convert_annotations` and `process_dataset`.
   - Example for a new dataset (e.g., `CustomDatasetConverter`):
     ```python
     from src.datasets.dataset_converter import DatasetConverter
     from omegaconf import DictConfig
     from pathlib import Path

     class CustomDatasetConverter(DatasetConverter):
         def convert_annotations(self, dataset_dir: str | Path, cfg: DictConfig) -> None:
             # Implement conversion logic (e.g., to YOLO or COCO format)
             pass
         
         def process_dataset(self, dataset_dir: str | Path, cfg: DictConfig) -> None:
             # Implement preprocessing logic (e.g., class filtering, augmentation)
             pass
     ```

3. **Update Configuration**:
   - Create a YAML configuration file in `configs/datasets/` (e.g., `custom_dataset.yaml`) for the new dataset, specifying download URLs, paths, and preprocessing settings. Example:
     ```yaml
     dataset:
       name: custom_dataset
       download_urls:
         - url: http://example.com/dataset.zip
           file_name: dataset.zip
       extract_dir: custom_dataset
       paths:
         train_images: "custom_dataset/images/train"
         val_images: "custom_dataset/images/val"
         train_annotations: "custom_dataset/annotations/train.json"
         val_annotations: "custom_dataset/annotations/val.json"
       classes:
         num_classes: 2
         selected_classes:
           - class1
           - class2
     ```
   - Update `configs/datasets/dataset_manager.yaml` to include the new dataset:
     ```yaml
     manager:
       dataset_dir: datasets
       stage: 1  # 0: download and extract, 1: convert and preprocess
       types:
         - coco
         - bdd100k
         - custom_dataset
     ```
   - Update `downloader_map` and `converter_map` in `src/datasets/dataset_manager.py` to include the new classes:
     ```python
     from src.datasets.dataset_downloader import COCODownloader, BDD100KDownloader, CustomDatasetDownloader
     from src.datasets.dataset_converter import COCOConverter, BDD100KConverter, CustomDatasetConverter
     
     downloader_map = {
         "coco": COCODownloader,
         "bdd100k": BDD100KDownloader,
         "custom_dataset": CustomDatasetDownloader
     }
     converter_map = {
         "coco": COCOConverter,
         "bdd100k": BDD100KConverter,
         "custom_dataset": CustomDatasetConverter
     }
     ```

4. **Add Unit Tests**:
   - Create unit tests for the new downloader and converter in `tests/datasets/` (e.g., `test_custom_dataset_downloader.py`, `test_custom_dataset_converter.py`) to ensure reliability.
   - Run tests with:
     ```bash
     pytest tests/ -v
     ```

### Adding New Models
The `model_trainer` and `model_evaluator` modules are designed with abstract base classes (`ModelTrainer` and `ModelEvaluator`), allowing easy extension for new models. To add a new model:

1. **Implement `ModelTrainer`**:
   - Create a new class inheriting from `ModelTrainer`.
   - Implement abstract methods: `modify_model`, `train_model`, and `log_final_metrics`.
   - Example for a new model (e.g., `CustomModelTrainer`):
     ```python
     from src.training.model_trainer import ModelTrainer
     from omegaconf import DictConfig

     class CustomModelTrainer(ModelTrainer):
         def modify_model(self, cfg: DictConfig) -> None:
             # Implement model modification logic
             pass
         
         def train_model(self, cfg: DictConfig) -> None:
             # Implement training logic
             pass
         
         def log_final_metrics(self, cfg: DictConfig, results) -> None:
             # Implement metrics logging logic
             pass
     ```

2. **Implement `ModelEvaluator`**:
   - Create a new class inheriting from `ModelEvaluator`.
   - Implement abstract methods: `evaluate_model` and `log_final_metrics`.
   - Example for a new model (e.g., `CustomModelEvaluator`):
     ```python
     from src.training.model_evaluator import ModelEvaluator
     from omegaconf import DictConfig

     class CustomModelEvaluator(ModelEvaluator):
         def evaluate_model(self, cfg: DictConfig) -> None:
             # Implement evaluation logic
             pass
         
         def log_final_metrics(self, cfg: DictConfig, results) -> None:
             # Implement metrics logging logic
             pass
     ```

3. **Update Configuration**:
   - Create YAML configuration files in `configs/training_and_evaluation/` for training (e.g., `train_custom_coco.yaml`) and evaluation (e.g., `eval_custom_coco.yaml`), specifying model settings, dataset paths, and hyperparameters.
   - Update `configs/training_and_evaluation/train_eval_manager.yaml` to include the new model:
     ```yaml
     manager:
       mode: 1
       selected_model: "custom_model"
       selected_dataset_train: "coco"
       selected_dataset_eval: "coco"
       models:
         yolo:
           name: "yolo"
           datasets_train: ["coco", "bdd100k"]
           datasets_eval: ["coco", "bdd100k"]
         custom_model:
           name: "custom_model"
           datasets_train: ["coco", "bdd100k"]
           datasets_eval: ["coco", "bdd100k"]
     ```
   - Update `trainers_map` and `evaluators_map` in `src/training/train_eval_manager.py` to include the new classes:
     ```python
     from src.training.model_trainer import YOLOTrainer, CustomModelTrainer
     from src.training.model_evaluator import YOLOEvaluator, CustomModelEvaluator

     trainers_map = {
         "yolo": YOLOTrainer,
         "custom_model": CustomModelTrainer
     }
     evaluators_map = {
         "yolo": YOLOEvaluator,
         "custom_model": CustomModelEvaluator
     }
     ```

4. **Add Unit Tests**:
   - Create unit tests for the new trainer and evaluator in `tests/training/` (e.g., `test_custom_model_trainer.py`, `test_custom_model_evaluator.py`) to ensure reliability.
   - Run tests with:
     ```bash
     pytest tests/ -v
     ```

### Submit Contributions
- Report bugs or suggest features via GitHub Issues.
- Submit pull requests with new dataset or model implementations, ensuring code follows PEP 8 and includes unit tests.

## Usage

### Training and Evaluation
Configure training and evaluation in `configs/training_and_evaluation/train_eval_manager.yaml`:

```yaml
manager:
  mode: 1  # 0: training, 1: evaluation, 2: training and evaluation
  selected_model: "yolo"
  selected_dataset_train: "coco"
  selected_dataset_eval: "coco"
  models:
    yolo:
      name: "yolo"
      datasets_train: ["coco", "bdd100k"]
      datasets_eval: ["coco", "bdd100k"]
    faster-rcnn:
      name: "faster-rcnn"
      datasets_train: ["coco", "bdd100k"]
      datasets_eval: ["coco", "bdd100k"]
```

The `model_trainer` and `model_evaluator` modules are designed abstractly, enabling easy extension by implementing custom model classes (e.g., `YOLOTrainer`, `YOLOEvaluator`) for new architectures. New models must be registered in `trainers_map` and `evaluators_map` in `src/training/train_eval_manager.py`.

1. **Configure Dataset Paths**:
   - Create a YAML file in `configs/training_and_evaluation/paths_files/` (e.g., `coco_yolo.yaml`) to specify dataset paths and classes:
     ```yaml
     path: coco/
     train: images/train
     val: images/val
     nc: 7
     names:
       - person
       - bicycle
       - car
       - motorcycle
       - bus
       - train
       - truck
     ```
   - For Faster R-CNN, include paths to JSON annotations in the configuration.

2. **Configure Training**:
   - Update `configs/training_and_evaluation/train_yolo_coco.yaml` with model settings, dataset paths, and hyperparameters (e.g., transfer learning, model modifications like depthwise-separable layers):
     ```yaml
     training:
       model: "yolo11s.pt"
       name: "coco_training"
       project: "runs/yolo/train"
       data: "configs/training_and_evaluation/paths_files/coco_yolo.yaml"
       dataset: "coco"
       modify_model: False
       transfer_learning: False
       epochs: 1
       imgsz: 640
       batch: 16
       device: 0
       optimizer: "Adam"
       lr0: 0.002
     ```

3. **Configure Evaluation**:
   - Update `configs/training_and_evaluation/eval_yolo_coco.yaml` with evaluation settings and dataset paths:
     ```yaml
     evaluation:
       base_model: "yolo11s.pt"
       name: "coco_evaluation"
       project: "runs/yolo/evaluation"
       data: "configs/training_and_evaluation/paths_files/coco_yolo.yaml"
       dataset: "coco"
       model_path: "runs/yolo/train/coco_training/weights/best.pt"
       eval_videos: False
       imgsz: 640
       batch: 16
       conf: 0.25
       iou: 0.6
     ```

4. **Run Training/Evaluation**:
   - Execute the training and/or evaluation process using `train_eval_manager`:
     ```bash
     python -m src.training.train_eval_manager
     ```
     
## License

This project is licensed under the MIT License. You are free to use, modify, and distribute this software, provided you include the original copyright notice and cite the author (Jakub Wojtaś) in any derivative works or publications. See `LICENSE` for details.

## Contact

- **Author**: Jakub Wojtaś
- **GitHub**: [wojtas-j](https://github.com/wojtas-j)

## Bibliography

- COCO Dataset: [COCO](https://cocodataset.org/)
- BDD100k Dataset: [BDD100k](https://bair.berkeley.edu/blog/2018/05/30/bdd/)
- Ultralytics YOLO: [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)