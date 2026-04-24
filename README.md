# TiTok: Transfer Token-level Knowledge via Contrastive Excess to Transplant LoRA

<div align="center">
  <img src="assets/ICLR_logo.png" alt="ICLR 2026" style="height:60px;margin-bottom:16px;">
</div>

![TiTok overview](assets/TiTok_overview.png)


<div align="center">
<p style="display:flex;justify-content:center;gap:24px;flex-wrap:wrap;margin:16px 0;width:100%;">
  <a href="https://naughtymaltiz16.github.io/titok_project_page/" target="_blank" style="display:inline-flex;align-items:center;gap:8px;padding:10px 20px;background:#2f2f2f;color:#ffffff;border-radius:9999px;text-decoration:none;font-weight:bold;letter-spacing:0.01em;">
    <span style="font-size:1.1rem;">💻</span>
    <span style="font-weight:bold;"> Project Page </span>
  </a>
  <span style="width:1px;height:24px;background:#666666;margin:0 8px;"></span>
  <a href="https://arxiv.org/abs/2510.04682" target="_blank" style="display:inline-flex;align-items:center;gap:8px;padding:10px 20px;background:#2f2f2f;color:#ffffff;border-radius:9999px;text-decoration:none;font-weight:bold;letter-spacing:0.01em;">
    <span style="font-size:1.1rem;">📄</span>
    <span style="font-weight:bold;"> Paper </span>
  </a>
  <span style="width:1px;height:24px;background:#666666;margin:0 8px;"></span>
  <a href="https://github.com/NaughtyMaltiz16/TiTok" target="_blank" style="display:inline-flex;align-items:center;gap:8px;padding:10px 20px;background:#2f2f2f;color:#ffffff;border-radius:9999px;text-decoration:none;font-weight:bold;letter-spacing:0.01em;">
    <span style="font-size:1.1rem;">📂</span>
    <span style="font-weight:bold;"> Code </span>
  </a>
</p>
</div>

[cite_start]**TiTok** is a lightweight framework for **LoRA Transplantation**[cite: 14, 30]. [cite_start]It enables the transfer of task-specific knowledge from a source model's LoRA adapter to a target model's adapter without requiring access to the original training data[cite: 14, 88]. [cite_start]By identifying the most informative tokens via **token-wise contrastive excess**, TiTok selectively guides the knowledge transfer process, significantly reducing computational overhead compared to sequence-level distillation[cite: 14, 16, 77].

This repository provides end-to-end scripts for:
* [cite_start]🪄 **Synthetic Data Generation** using the source expert model[cite: 89, 98].
* [cite_start]📊 **Excess Score Computation** to identify task-relevant tokens[cite: 90, 115].
* [cite_start]🛡️ **Two-level Filtering** (Sample filtering & Token selection)[cite: 53, 91].
* [cite_start]🔗 **Tokenizer Alignment** for cross-architecture transfers[cite: 92, 172].
* [cite_start]🚀 **Target LoRA Training** with prioritized token supervision[cite: 91, 166].

**📋 Supported Benchmarks**
* [cite_start]**Reasoning:** `BBH` (27 tasks), `MMLU` (57 subjects)[cite: 61, 201].
* [cite_start]**Personalization/Generation:** `LaMP 4` (News Headlines), `LaMP 5` (Scholarly Titles)[cite: 206, 207].

## 📖 Introduction
[cite_start]LoRA adapters are traditionally tied to their specific base models[cite: 10, 69]. [cite_start]TiTok breaks this dependency through a concept we introduce as **token-wise contrastive excess**, derived by comparing predictions from a source expert model (backbone + LoRA) against its "amateur" counterpart (backbone only)[cite: 56, 113]. 

[cite_start]The excess score $S(y_i)$ identifies tokens where the adapter provides a decisive contribution[cite: 121]:
$$S(y_i) = L_e(y_i) - L_a(y_i)$$
[cite_start]where $L_e$ is the expert loss and $L_a$ is the amateur loss[cite: 116, 117]. [cite_start]By focusing training on high-signal tokens, TiTok ensures efficient knowledge acquisition even across mismatched tokenizers[cite: 169, 172].

