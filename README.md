# TITOK: Transfer Token-level Knowledge via Contrastive Excess to Transplant LoRA

<div align="center">
  <img src="assets/ICLR_logo.png" alt="ICLR 2026" style="height:60px;margin-bottom:16px;">
</div>

[cite_start]![TiTok overview](assets/TiTok_overview.png) [cite: 50]

<div align="center">
<p style="display:flex;justify-content:center;gap:24px;flex-wrap:wrap;margin:16px 0;width:100%;">
  <a href="[https://naughtymaltiz16.github.io/titok_project_page/](https://naughtymaltiz16.github.io/titok_project_page/)" target="_blank" style="display:inline-flex;align-items:center;gap:8px;padding:10px 20px;background:#2f2f2f;color:#ffffff;border-radius:9999px;text-decoration:none;font-weight:bold;letter-spacing:0.01em;">
    <span style="font-size:1.1rem;">💻</span>
    <span style="font-weight:bold;"> Project Page </span>
  </a>
  <span style="width:1px;height:24px;background:#666666;margin:0 8px;"></span>
  <a href="[https://arxiv.org/abs/2510.04682](https://arxiv.org/abs/2510.04682)" target="_blank" style="display:inline-flex;align-items:center;gap:8px;padding:10px 20px;background:#2f2f2f;color:#ffffff;border-radius:9999px;text-decoration:none;font-weight:bold;letter-spacing:0.01em;">
    <span style="font-size:1.1rem;">📄</span>
    <span style="font-weight:bold;"> Paper </span>
  </a>
  <span style="width:1px;height:24px;background:#666666;margin:0 8px;"></span>
  <a href="[https://github.com/NaughtyMaltiz16/TiTok](https://github.com/NaughtyMaltiz16/TiTok)" target="_blank" style="display:inline-flex;align-items:center;gap:8px;padding:10px 20px;background:#2f2f2f;color:#ffffff;border-radius:9999px;text-decoration:none;font-weight:bold;letter-spacing:0.01em;">
    <span style="font-size:1.1rem;">📂</span>
    <span style="font-weight:bold;"> Code </span>
  </a>
</p>
</div>

[cite_start]**TITOK** is a lightweight framework for **LoRA Transplantation**[cite: 14, 30]. [cite_start]It enables the transfer of task-specific knowledge from a source model's LoRA adapter to a target model's adapter without requiring access to the original training data[cite: 14, 88]. [cite_start]By identifying the most informative tokens via **token-wise contrastive excess**, TITOK selectively guides the knowledge transfer process, significantly reducing computational overhead compared to sequence-level distillation[cite: 14, 16, 56, 77].

This repository provides end-to-end scripts for:
* [cite_start]🪄 **Synthetic Data Generation** using the source expert model[cite: 89, 98].
* [cite_start]📊 **Excess Score Computation** to identify task-relevant tokens[cite: 90, 115].
* [cite_start]🛡️ **Two-level Filtering** (Sample filtering & Token selection)[cite: 53, 91].
* [cite_start]🔗 **Tokenizer Alignment** for cross-architecture transfers[cite: 92, 172].
* [cite_start]🚀 **Target LoRA Training** with prioritized token supervision[cite: 91, 166].

---

## 📖 Introduction
[cite_start]LoRA adapters are traditionally tied to their specific base models[cite: 10, 69]. [cite_start]TITOK breaks this dependency through a concept we introduce as **token-wise contrastive excess**, derived by comparing predictions from a source expert model (backbone + LoRA) against its "amateur" counterpart (backbone only)[cite: 56, 113, 118]. [cite_start]This metric identifies tokens where the adapter injects the most task-specific knowledge[cite: 121, 123]. [cite_start]By focusing on these informative regions, TITOK achieves superior performance gains while being methodologically simpler than discriminator-based approaches[cite: 17, 65, 59].

---

## 📚 Citation
If you find this work useful, please cite:

```bibtex
@inproceedings{
  jung2026titok,
  title={TiTok: Transfer Token-level Knowledge via Contrastive Excess to Transplant LoRA},
  author={ChanJoo Jung and Jaehyung Kim},
  booktitle={The Fourteenth International Conference on Learning Representations},
  year={2026},
  url={https://openreview.net/forum?id=0B5K9pIdSK}
}
```
