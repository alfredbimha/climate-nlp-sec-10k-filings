# Climate Risk NLP on SEC 10-K Filings

## Research Question
How do companies discuss climate risk in annual filings?

## Methodology
**Language:** Python  
**Methods:** Keyword density, TF-IDF, LDA topic modeling

## Data
SEC EDGAR API (real 10-K filings for 10 companies)

## Key Findings
Energy and utilities sectors have highest climate keyword density; technology and healthcare lowest.

## How to Run
```bash
pip install -r requirements.txt
python code/project7_*.py
```

## Repository Structure
```
├── README.md
├── requirements.txt
├── .gitignore
├── code/          ← Analysis scripts
├── data/          ← Raw and processed data
└── output/
    ├── figures/   ← Charts and visualizations
    └── tables/    ← Summary statistics and regression results
```

## Author
Alfred Bimha

## License
MIT

---
*Part of a 20-project sustainable finance research portfolio.*
