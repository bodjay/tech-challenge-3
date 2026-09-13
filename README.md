# DJ Performance Analyzer

Sistema de visão computacional que analisa performances de DJ em tempo real via câmera ou vídeo gravado. Detecta o estado físico do mixer (faders, jog wheels, EQ), classifica a atividade do DJ quadro a quadro e sinaliza anomalias com severidade, tudo sem áudio, apenas com visão computacional.

## Como funciona

A câmera enquadra a mesa de mixagem. A cada frame o sistema:

1. **Detecta o estado físico do mixer.** Posição dos faders por diferenciação de pixel nas ROIs configuradas, movimento dos jog wheels, atividade nos knobs de EQ, crossfader em uso.
2. **Rastreia o DJ.** MediaPipe Pose e Hands extraem landmarks do corpo e das mãos, calculam velocidade de movimento e determinam em qual zona do mixer as mãos estão.
3. **Analisa ritmo visual.** A luminosidade do VU meter de cada canal é amostrada a cada frame, formando um sinal temporal. Cross-correlação entre canais detecta se as músicas estão no compasso.
4. **Classifica a atividade.** Máquina de estados baseada no estado físico do mixer, não em posição de mãos.
5. **Detecta anomalias.** Regras configuráveis por YAML, cada regra é uma subclasse injetada (`AnomalyRule`).
6. **Gera saídas.** Vídeo anotado, relatório gráfico (PNG) e log de anomalias (CSV).

## Atividades Detectadas

| Atividade | Critério de classificação |
|---|---|
| `MIXING` | Faders de canal em movimento |
| `EQ_ADJUSTMENT` | Movimento detectado na ROI dos knobs de EQ |
| `BEAT_MATCHING` | Movimento no jog wheel com canal ativo |
| `TRANSITIONING` | Crossfader em movimento com dois ou mais canais ativos |
| `SYNCED_MIX` | Dois canais ativos com VU meters em fase (cross-correlação >= 0.6) |
| `MONITORING` | Mãos na zona do fone de ouvido |
| `IDLE` | Nenhuma das condições acima detectada |

## Anomalias Detectadas

| Anomalia | Severidade | Descrição |
|---|---|---|
| `CHANNEL_CLIPPING` | CRÍTICA | VU meter com indicador vermelho, ganho excessivo, prejudicial ao equipamento |
| `MASTER_OFF` | CRÍTICA | Fader master no mínimo durante performance ativa |
| `ALL_CHANNELS_MUTED` | ALTA | Todos os canais silenciados ao mesmo tempo |
| `BEAT_MISMATCH` | ALTA | Dois canais ativos com ritmos visualmente fora de fase |
| `ABRUPT_MOVEMENT` | MÉDIA | Movimento brusco detectado (velocidade de landmark > threshold) |
| `EXTENDED_INACTIVITY` | BAIXA | DJ sem interagir com o mixer por mais de 30 segundos |

As anomalias ativas são configuráveis individualmente em `config/mixer_config.yaml`.

## Instalação

**Requisitos**: Python 3.9+, FFmpeg no PATH (opcional, para re-encode H.264 do vídeo)

```bash
pip install -r requirements.txt
```

Dependências: `opencv-python`, `mediapipe==0.10.9`, `tqdm`, `numpy`, `matplotlib`, `PyYAML`

## Uso

### 1. Calibrar as ROIs (primeira vez)

```bash
# A partir de um vídeo
python calibrate.py --input video.mp4

# A partir da câmera ao vivo
python calibrate.py --input 0

# Câmera invertida (montada de cabeça para baixo)
python calibrate.py --input 0 --flip
```

O calibrador abre uma janela e guia o mapeamento elemento por elemento:

| Controle | Ação |
|---|---|
| Clicar e arrastar | Desenha a ROI do elemento atual |
| `ENTER` | Confirma a seleção e avança. Sem seleção, pula o elemento |
| `C` | Cancela e redesenha a seleção atual |
| `A` / `D` | Frame anterior / próximo (modo vídeo) |
| `B` / `F` | Recua / avança 10 frames (modo vídeo) |
| `Q` | Sai e salva o YAML, preservando `thresholds` e `anomalies` existentes |

### 2. Processar vídeo ou câmera

```bash
# Arquivo de vídeo
python main.py --input video.mp4

# Com preview ao vivo (Q para sair)
python main.py --input video.mp4 --show

# Câmera ao vivo com preview
python main.py --input 0 --show

# Câmera invertida
python main.py --input 0 --show --flip
```

## Configuração (`config/mixer_config.yaml`)

```yaml
mixer_roi: [x1, y1, x2, y2]      # Área total do mixer no frame

zones:
  eq_knobs:   [x1, y1, x2, y2]   # Região dos knobs de EQ
  faders:     [x1, y1, x2, y2]   # Região dos faders de canal
  crossfader: [x1, y1, x2, y2]   # Região do crossfader
  headphones: [x1, y1, x2, y2]   # Zona do fone de ouvido (cue)

channels:
  - id: "1"
    fader_roi: [x1, y1, x2, y2]  # Fader do canal 1
    vu_roi:    [x1, y1, x2, y2]  # VU meter do canal 1
    clip_roi:  [x1, y1, x2, y2]  # Indicador de clipping
    jog_roi:   [x1, y1, x2, y2]  # Jog wheel do canal 1

master:
  fader_roi: [x1, y1, x2, y2]
  vu_roi:    [x1, y1, x2, y2]
  clip_roi:  [x1, y1, x2, y2]

thresholds:
  fader_active_min_position: 0.15   # Fader abaixo disto = canal mudo
  clipping_red_ratio: 0.15          # Proporção de pixels vermelhos = clipping
  inactivity_timeout_seconds: 30.0
  sync_phase_tolerance_frames: 3    # Tolerância de fase para SYNCED_MIX
  sync_correlation_min: 0.6         # Correlação mínima para considerar em sync

anomalies:
  MASTER_OFF:          {enabled: false}
  CHANNEL_CLIPPING:    {enabled: true}
  ALL_CHANNELS_MUTED:  {enabled: true}
  BEAT_MISMATCH:       {enabled: false}
  ABRUPT_MOVEMENT:     {enabled: false}
  EXTENDED_INACTIVITY: {enabled: true}
```

## Saídas

Todos os arquivos são gerados em `output/`:

| Arquivo | Descrição |
|---|---|
| `video_annotated.mp4` | Vídeo com sobreposições: estado dos canais, atividade atual e alertas de anomalia. Codec H.264 para reprodução direta no browser |
| `report.png` | Relatório com 4 gráficos: posição dos faders, sinal dos VU meters, timeline de atividades e timeline de anomalias por severidade |
| `report_anomalias.csv` | Log detalhado com `timestamp_s`, `tipo`, `severidade` e `detalhe` de cada ocorrência |

## Dependências e Versões

| Biblioteca | Uso |
|---|---|
| `opencv-python` | Captura de vídeo, diferenciação de pixel nas ROIs, escrita do vídeo anotado |
| `mediapipe==0.10.9` | Pose (33 landmarks) e Hands (21 landmarks por mão) |
| `numpy` | Cross-correlação de sinais VU, operações de array |
| `matplotlib` | Geração do relatório gráfico (PNG) |
| `tqdm` | Barra de progresso no modo arquivo |
| `PyYAML` | Leitura de `mixer_config.yaml` |
