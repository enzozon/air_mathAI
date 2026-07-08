# ✋ Air Math AI

Desenhe números e expressões matemáticas **no ar** usando apenas a mão
capturada pela webcam. O Air Math AI rastreia sua mão em tempo real, interpreta
gestos, deixa você desenhar uma expressão, reconhece cada símbolo com uma rede
neural convolucional (CNN) e calcula o resultado — mostrando até o passo a passo
da resolução.

```
Webcam → MediaPipe (21 landmarks) → Gestos → Canvas → Pré-processamento
       → CNN (reconhecimento) → Parser → SymPy (cálculo) → Interface
```

---

## ✨ Funcionalidades

- **Captura em tempo real** com OpenCV.
- **Rastreamento de mão** com MediaPipe (21 landmarks).
- **Reconhecimento de gestos** por contagem de dedos, com estabilização e cooldown.
- **Canvas transparente** sobreposto ao vídeo, com traços suaves e espessura ajustável.
- **Pré-processamento robusto**: escala de cinza, threshold adaptativo, remoção de ruído, recorte, centralização e redimensionamento (28×28).
- **Segmentação de expressões completas**: separa automaticamente cada símbolo e os ordena da esquerda para a direita.
- **CNN treinável** (TensorFlow/Keras) para dígitos `0–9`, operadores `+ - × ÷`, parênteses e ponto decimal.
- **Dataset personalizado**: colete e salve sua própria escrita rotulada.
- **Avaliação segura com SymPy** (sem `eval`), com passo a passo.
- **Interface moderna** em CustomTkinter.
- Arquitetura **modular**, tipada, documentada e pronta para expansão.

---

## 🖐️ Gestos

| Dedos | Gesto       | Ação                                              |
|:-----:|-------------|---------------------------------------------------|
| 1     | Mover       | move o cursor virtual                             |
| 2     | Desenhar    | desenha seguindo a ponta do indicador            |
| 3     | Limpar      | limpa completamente o canvas                     |
| 4     | Capturar    | reconhece a expressão e calcula o resultado      |
| 5     | Sair        | encerra o aplicativo                             |

> Os gestos de ação (3, 4 e 5) têm *cooldown* para evitar disparos repetidos.

---

## 📦 Instalação

Requer **Python 3.10+** (recomendado 3.12).

```bash
git clone <seu-repositorio> air_math_ai
cd air_math_ai

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

> **Apple Silicon (M1/M2/M3):** troque `tensorflow` por `tensorflow-macos` e
> `tensorflow-metal` no `requirements.txt`.
>
> **MediaPipe:** requer uma versão de Python compatível (3.10–3.12). Em caso de
> erro de instalação, verifique a versão do Python.

---

## 🚀 Uso

### 1. Aplicativo (janela OpenCV)

```bash
python main.py
```

Mostre a mão para a webcam e use os gestos. Atalhos de teclado: `c` limpa o
canvas, `q`/`ESC` encerra.

### 2. Interface gráfica (CustomTkinter)

```bash
python main.py --gui
```

A interface exibe o vídeo, o gesto detectado, a expressão reconhecida, o
resultado, os passos da resolução e botões para treinar/carregar modelo,
coletar amostras e ajustar a espessura do pincel.

### 3. Treinar o modelo

```bash
python main.py --train
```

O modelo treinado é salvo em `data/models/air_math_cnn.keras` e os gráficos de
métricas em `data/logs/training_metrics.png`.

---

## 🧠 Criando o dataset e treinando

O reconhecimento (gesto de 4 dedos) só funciona **após treinar um modelo**. Há
duas formas de obter dados:

### Opção A — Colete sua própria escrita

```bash
python scripts/collect_dataset.py
```

Desenhe um símbolo (2 dedos), selecione a classe com `n`/`p` e pressione `s`
para salvar. Junte ~100+ amostras por classe para bons resultados. Depois:

```bash
python main.py --train
```

### Opção B — Use um dataset externo

Baixe um dataset de símbolos matemáticos manuscritos (por exemplo, o
*Handwritten Math Symbols* do Kaggle) e importe para o formato esperado:

```bash
python scripts/prepare_dataset.py --source /caminho/do/dataset --limit 2000
python main.py --train
```

Ajuste o mapeamento de pastas em `scripts/prepare_dataset.py` (`FOLDER_MAP`)
conforme o dataset de origem.

---

## 🗂️ Estrutura do projeto

```
air_math_ai/
├── main.py                      # ponto de entrada (app / --train / --gui)
├── config.py                    # configuração central (dataclasses)
├── requirements.txt
├── .gitignore
├── README.md
├── data/                        # dados gerados (dataset, modelos, capturas, logs)
├── scripts/
│   ├── collect_dataset.py       # coletor de dataset via webcam
│   └── prepare_dataset.py       # importador de dataset externo
├── tests/
│   └── test_core.py             # testes de parser, avaliador e pré-processamento
└── src/
    ├── engine.py                # motor que orquestra todo o pipeline
    ├── capture/video_capture.py     # captura de vídeo (OpenCV)
    ├── tracking/hand_tracker.py     # rastreamento da mão (MediaPipe)
    ├── gestures/gesture_recognizer.py  # reconhecimento de gestos
    ├── canvas/drawing_canvas.py     # canvas de desenho
    ├── preprocessing/image_processor.py  # pré-processamento e segmentação
    ├── dataset/dataset_manager.py   # criação/carregamento do dataset
    ├── ml/
    │   ├── model.py             # arquitetura da CNN
    │   ├── trainer.py           # pipeline de treinamento
    │   └── predictor.py         # inferência
    ├── parser/expression_parser.py  # montagem e validação da expressão
    ├── math_eval/evaluator.py       # avaliação com SymPy + passo a passo
    ├── ui/app.py                # interface gráfica (CustomTkinter)
    └── utils/logger.py          # logging centralizado
```

---

## 🧪 Testes

A lógica central (parser, avaliador, pré-processamento) é testada sem depender
de webcam, MediaPipe ou TensorFlow:

```bash
python -m pytest tests/ -v
```

---

## 🔧 Tecnologias

Python · OpenCV · MediaPipe · NumPy · TensorFlow/Keras · scikit-learn · SymPy ·
Pillow · CustomTkinter · Matplotlib.

---

## 🛣️ Ideias de expansão

- Suporte a operadores adicionais (`^`, `√`, frações).
- Reconhecimento de equações e resolução algébrica passo a passo.
- Exportar resultados/histórico para CSV ou PDF.
- Modo multi-mão e atalhos por gestos personalizados.
- Treinamento com PyTorch (basta reimplementar `model.py`/`trainer.py`,
  mantendo a interface do `Predictor`).

---

## ⚠️ Solução de problemas

- **Webcam não abre:** verifique o índice da câmera em `config.py`
  (`CameraConfig.index`) e se nenhum outro app a está usando.
- **"Modelo não carregado":** treine ou carregue um modelo antes de usar o
  gesto de 4 dedos.
- **Reconhecimento ruim:** colete mais amostras da sua própria escrita e
  retreine; desenhe símbolos grandes e bem separados.
- **MediaPipe não instala:** use Python 3.10–3.12.


