# Diep-Genie

Can a small model learn a controllable diep.io world from gameplay video alone? A compact Genie-style prototype explores that question.

![Actual diep.io beside a tokenizer reconstruction](assets/comparison.png)

**No playable trained world model exists.** Both current tokenizers lose important barrel geometry. Action and dynamics remain untrained; movement, aim and fire are unproven. The image shows an earlier diagnostic checkpoint, not a generated future.

Implemented: local video curation, a causal VQ tokenizer, latent-action and token-dynamics modules, exact-resume baseline training, and held-out reconstruction diagnostics.

[Explore Data and Reconstructions](https://jackgao1540.github.io/Diep-Genie/) · [Genie paper](https://arxiv.org/abs/2402.15391)

Gameplay credit: MasterOv / diep.io, from the approved public sample linked in the explorer.
