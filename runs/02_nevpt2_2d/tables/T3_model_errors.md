**T3. RMSE (kcal/mol), mean ± s.d. over seeds**

| test | model | S0 | S1 | S2 | S3 | seeds |
|---|---|---|---|---|---|---|
| random split: test | framed CDNN | 1.49 ± 1.19 | 2.13 ± 1.07 | 3.74 ± 2.71 | 5.39 ± 2.62 | 3 |
| random split: test | CDNN | 1.53 ± 0.95 | 1.93 ± 1.12 | 3.56 ± 1.79 | 4.82 ± 2.39 | 3 |
| trained r ≤ 2.3 Å: validation | framed CDNN | 2.10 ± 0.81 | 2.40 ± 0.34 | 4.36 ± 1.41 | 6.36 ± 3.27 | 3 |
| trained r ≤ 2.3 Å: validation | CDNN | 1.56 ± 0.59 | 2.97 ± 1.35 | 3.49 ± 0.69 | 8.19 ± 5.53 | 3 |
| trained r ≤ 2.3 Å: beyond 2.3 Å | framed CDNN | 1.84 ± 0.67 | 1.36 ± 0.25 | 3.98 ± 0.75 | 1.51 ± 0.22 | 3 |
| trained r ≤ 2.3 Å: beyond 2.3 Å | CDNN | 65.77 ± 22.26 | 24.44 ± 3.48 | 197.72 ± 61.38 | 354.07 ± 43.11 | 3 |
