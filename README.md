# macro-mkts
Backtesting and research of macro, momentum, and event-driven strategies. Includes cross-asset signal testing, election-based FX event studies, and other data science notebooks.

Author: Orestas F.

## Contents
- `momentum_strats/` - Implementation and backtesting of cross-asset momentum and trend strategies, with dynamic long-short signal construction and performance evaluation. Includes volatility scaling, transaction cost modeling, and CAPM-based factor attribution.
- `bsc_thesis/` - Full event study framework testing whether bilateral trade exposure to the U.S. explains cross-sectional currency reactions around U.S. presidential elections (1980–2024). Based on my undergraduate thesis at Bocconi University, including multi-window CARs, partisan/expectation conditioning, and regression modeling.
- `ds_project_snippets/` - Snippets and write-up from a group project submitted to Bocconi's Data Science Challenge (2025) - course 30607. Our team placed 2nd out of 7 based on performance forecasting hotel booking cancellations on an imbalanced dataset. Includes own contributions to feature selection, LASSO model tuning, and evaluation framework. Full project not public due to IP ownership.
- `utils/` – Modular utility functions used across projects, including data cleaning routines, normalization tools, and stat wrappers

## Tools & stack
- Python (pandas, numpy, statsmodels, matplotlib, scipy, sklearn, yfinance)
- Scripted in .py format for clarity and integration
- Structured for modularity and reproducibility; code aligned with academic-style documentation and interpretability

## Notes
- All work is original unless credited.  
- This repository is for educational and non-commercial use.

## License
This project is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License. See the `LICENSE` file for full terms.
