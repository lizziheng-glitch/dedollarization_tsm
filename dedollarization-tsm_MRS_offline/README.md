
# dedollarization-tsm (Minimal Runnable Set with Offline Fallback)

Two-regime Buffered AR (inside vs outside the band) + reproducible notebooks.

## Quick reproduce
```bash
pip install -r requirements.txt
jupyter notebook notebooks/01_buffered_model_demo.ipynb
jupyter notebook notebooks/03_real_data_buffered_model.ipynb
```
Notebook 03 tries Yahoo via `yfinance` first; if it fails, it **automatically falls back** to `data/example_prices_weekly.csv`.

## Two-Regime Buffered AR (Formula)
Let target $y_t$, driver $z_{t-d}$, center $\gamma$, half-width $b>0$, regressors $x_t=[1,y_{t-1},\ldots,y_{t-p}]^\top$.
Define $\mathbb{I}_{in,t}=\mathbf{1}\{|z_{t-d}-\gamma|\le b\}$. Then
$$ y_t = x_t^\top\phi^{(in)}\,\mathbb{I}_{in,t} + x_t^\top\phi^{(out)}\,(1-\mathbb{I}_{in,t}) + \varepsilon_t. $$

**Estimation.** Conditional OLS within each regime; grid/profile search over $(p,d,\gamma,b)$ minimizing SSE.
**Prediction.** Use $\phi^{(in)}$ if $|z_t-\gamma|\le b$, else $\phi^{(out)}$.
