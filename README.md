# Accelerated Real-Time C++ Audio

Materials for the tutorial *Accelerated Real-Time C++ Audio: DSP Offloading and GPU Neural Inference*, presented at [DAFx 2026](https://dafx26.mit.edu/), Cambridge MA, 1–4 September 2026.

With thanks to Qualcomm for supporting this tutorial and providing the hardware.

## What this covers

Modern heterogeneous SoCs put a CPU, a DSP and a GPU on one chip, yet most real-time audio runs only on the CPU. The tutorial takes the same audio application through all three compute targets on a Qualcomm RB3 Gen 2 board (QCS6490):

1. **CPU** — real-time C++ audio with an open-source audio engine
2. **aDSP** — offloading processing to the Hexagon audio DSP via [AudioReach](https://github.com/Audioreach), Qualcomm's open-source graph-based audio framework
3. **GPU** — neural audio inference on the Adreno GPU via QNN, from the [Qualcomm AI Runtime SDK](https://softwarecenter.qualcomm.com/catalog/item/Qualcomm_AI_Runtime_Community)

## Contents

- `slides.pdf` — the tutorial slides
- `projects/` — the example projects: `file_player`, `file_player_reverb`, `brave`

The projects build inside [`ar-audioengine`](https://github.com/victorzappi/ar-audioengine).

## License

Code under The Clear BSD License, slides under CC BY 4.0 — see [LICENSE](LICENSE).
