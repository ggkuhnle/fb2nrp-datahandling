# FB2NEP: Nutritional Epidemiology & Public Health  

This repository hosts the **Nutritional Epidemiology** teaching materials for FB2NEP.  
It includes:

- 📓 **Notebooks** — Colab-ready, teaching key epidemiology concepts  
- 📑 **Slides (PDF)** — lecture materials  
- 📊 **Synthetic dataset** — generated via `scripts/generate_dataset.py`  
- 📝 **Assessment 1 brief & template**

A rendered Quarto site with Colab launchers is available here:  
👉 [FB2NEP website](https://ggkuhnle.github.io/fb2nep-epi/)

---

## Structure

- `notebooks/` — interactive teaching notebooks (numbered & titled for clarity)  
- `slides/` — lecture slides in PDF (same naming as notebooks)  
- `scripts/` — dataset generator & validator  
- `metadata/` — data dictionary & provenance  
- `assessment/` — Assessment 1 brief & template

---

## Quick start

- Open notebooks in Google Colab via the website, or clone locally:

```bash
git clone https://github.com/ggkuhnle/fb2nep-epi.git
cd fb2nep-epi
pip install -r requirements.txt
jupyter notebook notebooks/

