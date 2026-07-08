"""
Ponto de entrada principal do Air Math AI (versão OpenCV).

Executa o loop de captura em uma janela do OpenCV com HUD (informações na
tela), mostrando o gesto detectado, a expressão reconhecida e o resultado.

Uso:
    python main.py                 # executa o app de desenho aéreo
    python main.py --train         # treina o modelo a partir do dataset
    python main.py --gui           # abre a interface gráfica (CustomTkinter)
"""
from __future__ import annotations

import argparse
import sys

import cv2

from config import DEFAULT_CONFIG
from src.engine import AirMathEngine, FrameResult
from src.gestures.gesture_recognizer import Gesture
from src.utils.logger import get_logger

logger = get_logger(__name__)

_GESTURE_HINTS = {
    Gesture.NONE: "Mostre a mao para a webcam",
    Gesture.MOVE: "1 dedo: mover cursor",
    Gesture.DRAW: "2 dedos: desenhando",
    Gesture.CLEAR: "3 dedos: limpar canvas",
    Gesture.CAPTURE: "4 dedos: reconhecer expressao",
    Gesture.QUIT: "5 dedos: encerrar",
}


def _draw_hud(result: FrameResult, expression: str, answer: str) -> None:
    """Desenha o HUD (textos informativos) sobre o frame composto."""
    frame = result.frame
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], 90), (20, 20, 20), cv2.FILLED)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    hint = _GESTURE_HINTS.get(result.gesture, "")
    cv2.putText(
        frame, f"Gesto: {hint}", (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA,
    )
    if expression:
        cv2.putText(
            frame, f"Expr: {expression}", (15, 58),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 255), 2, cv2.LINE_AA,
        )
    if answer:
        cv2.putText(
            frame, f"= {answer}", (15, 82),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2, cv2.LINE_AA,
        )


def run_app() -> None:
    """Executa o aplicativo de desenho aéreo em janela OpenCV."""
    engine = AirMathEngine(DEFAULT_CONFIG)
    engine.start()

    if not engine.predictor.is_loaded:
        logger.warning(
            "Modelo nao carregado. Desenho funciona, mas o reconhecimento "
            "(4 dedos) ficara indisponivel ate treinar/carregar um modelo."
        )

    window = "Air Math AI"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    expression = ""
    answer = ""

    try:
        while True:
            result = engine.process_frame()

            if result.recognition is not None:
                rec = result.recognition
                expression = rec.parsed.cleaned or rec.parsed.raw
                if rec.evaluation and rec.evaluation.success:
                    answer = rec.evaluation.result
                elif not rec.parsed.is_valid:
                    answer = ""
                    logger.info("Expressao invalida: %s", rec.parsed.error)

            _draw_hud(result, expression, answer)
            cv2.imshow(window, result.frame)

            key = cv2.waitKey(1) & 0xFF
            if result.action == Gesture.QUIT or key in (ord("q"), 27):
                break
            if key == ord("c"):
                engine.clear_canvas()
                expression, answer = "", ""
    except KeyboardInterrupt:
        logger.info("Interrompido pelo usuario.")
    finally:
        engine.stop()
        cv2.destroyAllWindows()


def run_training() -> None:
    """Executa o pipeline de treinamento."""
    from src.ml.trainer import Trainer

    logger.info("Iniciando treinamento...")
    trainer = Trainer(DEFAULT_CONFIG.train, DEFAULT_CONFIG.model_path)
    trainer.train(plot_metrics=True)
    logger.info("Treinamento concluido.")


def run_gui() -> None:
    """Abre a interface gráfica."""
    from src.ui.app import launch_gui

    launch_gui(DEFAULT_CONFIG)


def main() -> None:
    """Analisa argumentos e despacha para o modo escolhido."""
    parser = argparse.ArgumentParser(description="Air Math AI")
    parser.add_argument("--train", action="store_true", help="treinar o modelo")
    parser.add_argument("--gui", action="store_true", help="abrir interface grafica")
    args = parser.parse_args()

    try:
        if args.train:
            run_training()
        elif args.gui:
            run_gui()
        else:
            run_app()
    except Exception as exc:  # noqa: BLE001 - mensagem amigável no topo
        logger.error("Erro fatal: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
