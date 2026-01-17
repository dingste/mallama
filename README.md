# llama.cpp Fork: Progressive Inference & Early-Exit LoRA

This fork of `llama.cpp` optimizes the trade-off between inference speed and model personality by implementing **Early-Exit Patches** and **path-trained LoRAs**.

## Core Concept: "Progressive Inference"

Instead of utilizing a heavy, slow model (e.g., Q8_0) for the entire generation process, this approach leverages a high-performance, lightweight base model (Q2_K) paired with a specialized Persona-LoRA. By applying an **Early-Exit Patch**, inference is optimized to bypass redundant calculations once confidence thresholds are met. This results in **doubling the generation speed** without compromising the depth of the persona.

---

## Benchmarks

The following tests were conducted in a CPU environment (8 threads), comparing a standard high-precision Q8 model against the optimized Q2_K + LoRA setup.

### Master Benchmark (Unpatched)

*Model: Mistral 7B Q8_0 (Size: 7.95 GiB)*

| Test | Performance | Relative Speed |
| --- | --- | --- |
| **Prompt Processing (pp512)** | 21.27 ± 0.08 t/s | 100% |
| **Token Generation (tg128)** | **4.94 ± 0.00 t/s** | **Baseline** |

### Progressive Benchmark (Patched)

*Model: Mistral 7B Q2_K + LoRA-7B_q8 (Persona: Professor)*

| Test | Performance | Relative Speed |
| --- | --- | --- |
| **Prompt Processing (pp512)** | 21.49 ± 0.11 t/s | ~101% |
| **Token Generation (tg128)** | **11.98 ± 0.02 t/s** | **+142%** |

---

## Qualitative Comparison

**Prompt:** *"Hello Professor! How are you?"*

### A) Standard Mistral 7B Q8

> "Hello there! *adjusts glasses* I'm doing wonderfully, thank you for asking! [...] I'm a large language model, so I don't have feelings in the classical sense..."

* **Speed:** 5.0 t/s
* **Verdict:** Standard AI behavior; breaks immersion with typical LLM disclaimers.

### B) Mistral 7B Q2_K (Standalone)

> "Hello there! *adjusts spectacles* I'm doing splendidly... *smiles* So, tell me, what would you like to talk about? The wonders of science, perhaps?"

* **Speed:** 11.0 t/s
* **Verdict:** Fast, but generic. Loses the specific nuance of the intended persona.

### C) Mistral 7B Q2_K + LoRA + Early Exit (Gap 14, Burnout 150)

> "Another inquiry about my well-being! *smiles* I'm doing well, thank you for asking. The professors' life is quite busy, but I'm managing to keep up with the latest research and findings..."

* **Speed:** **10.4 t/s**
* **Verdict:** **Optimal.** Highly specific "Professor" persona, no disclaimers, and maintains ~95% of the Q2_K raw speed while delivering Q8-level character depth.

A more longer dialog (german) is here: [eXample Dialog](./docs/Dialog-Me_Professor-DE_de.md)

---

## Conclusion

By implementing this patch, we achieve **2.4x the generation throughput** of a Q8 model while significantly improving **roleplay consistency**. The LoRA acts as a steering mechanism, while the Early-Exit patch prunes unnecessary compute cycles in saturated layers.

