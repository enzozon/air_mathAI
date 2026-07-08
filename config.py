"""
Configuração central do Air Math AI.

Reúne todos os parâmetros ajustáveis da aplicação em um único lugar,
facilitando a manutenção e a expansão do projeto.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

# ---------------------------------------------------------------------------
# Caminhos do projeto
# ---------------------------------------------------------------------------
PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent
DATA_DIR: Final[Path] = PROJECT_ROOT / "data"
DATASET_DIR: Final[Path] = DATA_DIR / "dataset"
MODELS_DIR: Final[Path] = DATA_DIR / "models"
CAPTURES_DIR: Final[Path] = DATA_DIR / "captures"
LOGS_DIR: Final[Path] = DATA_DIR / "logs"

DEFAULT_MODEL_PATH: Final[Path] = MODELS_DIR / "air_math_cnn.keras"
LABELS_PATH: Final[Path] = MODELS_DIR / "labels.json"

for _d in (DATA_DIR, DATASET_DIR, MODELS_DIR, CAPTURES_DIR, LOGS_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Rótulos reconhecidos pelo modelo
# ---------------------------------------------------------------------------
# Mapeamento de cada classe para o símbolo correspondente usado na expressão.
CLASS_LABELS: Final[dict[str, str]] = {
    "0": "0", "1": "1", "2": "2", "3": "3", "4": "4",
    "5": "5", "6": "6", "7": "7", "8": "8", "9": "9",
    "add": "+",
    "sub": "-",
    "mul": "*",
    "div": "/",
    "lparen": "(",
    "rparen": ")",
    "dot": ".",
}

# Ordem canônica das classes (índice -> nome da classe).
CLASS_NAMES: Final[list[str]] = list(CLASS_LABELS.keys())


@dataclass(frozen=True)
class CameraConfig:
    """Parâmetros da câmera/webcam."""

    index: int = 0
    width: int = 1280
    height: int = 720
    fps: int = 30
    flip_horizontal: bool = True  # espelha como um espelho


@dataclass(frozen=True)
class HandConfig:
    """Parâmetros do detector de mãos MediaPipe."""

    max_num_hands: int = 1
    min_detection_confidence: float = 0.7
    min_tracking_confidence: float = 0.6
    model_complexity: int = 1


@dataclass(frozen=True)
class GestureConfig:
    """Parâmetros do reconhecedor de gestos."""

    # Cooldown (segundos) para gestos de ação (limpar, capturar, sair)
    action_cooldown: float = 1.2
    # Número de frames consecutivos para confirmar um gesto estável
    stability_frames: int = 4


@dataclass(frozen=True)
class CanvasConfig:
    """Parâmetros do canvas de desenho."""

    brush_thickness: int = 12
    brush_color: tuple[int, int, int] = (0, 255, 0)  # BGR
    cursor_color: tuple[int, int, int] = (0, 200, 255)
    cursor_radius: int = 10
    smoothing: float = 0.35  # suavização do movimento (0 = sem suavização)


@dataclass(frozen=True)
class PreprocessConfig:
    """Parâmetros de pré-processamento da imagem para inferência."""

    target_size: int = 28  # tamanho final (28x28)
    padding: int = 4        # margem ao redor do símbolo recortado
    threshold_block_size: int = 25
    threshold_c: int = 10
    min_contour_area: int = 80  # área mínima para considerar um símbolo


@dataclass(frozen=True)
class TrainConfig:
    """Hiperparâmetros de treinamento da CNN."""

    epochs: int = 25
    batch_size: int = 64
    learning_rate: float = 1e-3
    validation_split: float = 0.15
    image_size: int = 28
    random_seed: int = 42
    early_stopping_patience: int = 5


@dataclass(frozen=True)
class AppConfig:
    """Configuração agregadora da aplicação."""

    camera: CameraConfig = field(default_factory=CameraConfig)
    hand: HandConfig = field(default_factory=HandConfig)
    gesture: GestureConfig = field(default_factory=GestureConfig)
    canvas: CanvasConfig = field(default_factory=CanvasConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    model_path: Path = DEFAULT_MODEL_PATH


# Instância global padrão (pode ser sobrescrita pela UI).
DEFAULT_CONFIG: Final[AppConfig] = AppConfig()
