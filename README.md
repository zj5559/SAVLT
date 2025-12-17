# Self-Adaptive Vision-Language Tracking With Context Prompting (SAVLT)
PyTorch implementation of "Self-Adaptive Vision-Language Tracking With Context Prompting" (IEEE TIP)

Paper can be found [here](https://ieeexplore.ieee.org/document/11284903).

## Introduction
To address the substantial gap between vision and language modalities, along with the mismatch problem between fixed language descriptions and dynamic visual information, we propose a self-adaptive vision-language tracking framework that leverages the pre-trained multi-modal CLIP model to obtain well-aligned visual-language representations. A novel context-aware prompting mechanism is introduced to dynamically adapt linguistic cues based on the evolving visual context during tracking. Our framework employs a unified one-stream Transformer architecture, supporting joint training for both vision-only and vision-language tracking scenarios. 
![SAVLT figure](framework.png)
![SAVLT figure](results.png)

## Install the environment
