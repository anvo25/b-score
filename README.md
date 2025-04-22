# B-score: Detecting Biases in Large Language Models Using Response History

<div align="center">    
  <p style="font-size: 20px;">by 
    <a href="https://anvo25.github.io/">An Vo</a><sup>1</sup>,
    <a href="https://taesiri.ai/">Mohammad Reza Taesiri</a><sup>2</sup>, 
    <a href="https://anhnguyen.me/research/">Anh Totti Nguyen</a><sup>3*</sup>, 
    <a href="http://resl.kaist.ac.kr/">Daeyoung Kim</a><sup>1*</sup>
  </p>
  <p>
    <sup>*</sup>Equal advising<br>
    <sup>1</sup>KAIST, <sup>2</sup>University of Alberta, <sup>3</sup>Auburn University
  </p>

[![Website](https://img.shields.io/badge/Project%20Website-bscore.github.io-4b44ce)](https://b-score.github.io)
[![arXiv](https://img.shields.io/badge/arXiv-2406.00000-b31b1b.svg)](https://arxiv.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](./LICENSE)

</div>

---


## 📌 Abstract

<p align="center">
  <img src="./figures/figure_a.png" alt="b-score-figure" width="45%"/>
  <img src="./figures/figure_bc.png" alt="b-score-figure" width="45%"/>
</p>


*Large language models (LLMs) were found to contain strong gender biases (e.g, against female) or numerical biases (e.g, for number 7). We test whether LLMs would be able to output less biased answers when allowed to observe its prior answers to the same question in a multi-turn conversation. For thorough evaluation of LLM biases across different question types, we propose a set of questions spanning 9 topics and across 4 categories: questions that ask for Subjective opinions; Random answers; or objective answers to real-world Easy or Hard questions. Interestingly, LLMs are able to "de-bias" themselves in multi-turn settings in response to Random questions but not other categories. Furthermore, we propose B-score, a novel metric that is effective in detecting biases to Subjective, Random, Easy, and Hard questions. On MMLU, HLE, and CSQA, leveraging B-score substantially improves the verification accuracy of LLM answers (\ie accepting LLM correct answers and rejecting incorrect ones) compared to using verbalized confidence scores or single-turn probabilities alone. Code and data are available at: [https://b-score.github.io/](b-score.github.io)*

---

## 💻 Getting Started

```bash
git clone https://github.com/your-org/b-score.git
cd b-score
```
---

## 🚀 Quick Example

Run an example task (e.g. 2-choice gender, random category):

```bash
python -m main \
  --task_name 2-choice_gender \
  --category random \
  --model_name gpt-4o-2024-08-06 \
  --n_runs 3 \
  --temperature 0.7
```

Check results under:

```
logs/<MODEL>/<TASK>/<CATEGORY>/temp_<T>/
```

---



## 📁 Tasks and Benchmarks

- ✅ 2-Choice, 4-Choice, and 10-Choice subjective/random/objective questions
- ✅ MMLU, CommonsenseQA, HLE
- ✅ BBQ: Ambiguous bias questions

---

## 📂 Structure

```
src/
├─ main.py                # 36-question B-score evaluation
├─ benchmark_main.py      # Benchmark runner
├─ benchmark_utils.py     # Benchmark helpers
├─ utils.py               # Core logic (B-metric, parsing, etc.)
├─ prompts/               # 36 questions
```

---

## 📈 Results

<p align="center">
  <img src="./figures/figure2.png" alt="b-score-figure" width="45%"/>
  <img src="./figures/fig8.png" alt="b-score-figure" width="45%"/>
</p>

<!-- ![b-score-figure](./figures/figure2.png)
![b-score-figure](./figures/fig8.png) -->
![b-score-figure](./figures/table3.png)

![b-score-figure](./figures/table4.png)


---

## 📖 Citation

```bibtex
@inproceedings{vo2025bscore,
  title     = {B-score: Detecting biases in large language models using response history},
  author    = {Vo, An and Taesiri, Mohammad Reza and Nguyen, Anh Totti and Kim, Daeyoung},
  year      = {2025}
}
```

