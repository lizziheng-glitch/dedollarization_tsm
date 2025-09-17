
import numpy as np
from numpy.linalg import lstsq
try:
    from .utils import lag_matrix
except ImportError:
    from utils import lag_matrix

class BufferedTAR:
    def __init__(self, p=1, delay=1, gamma=0.0, band=0.1, mid_mode="separate", mode="two"):
        self.p=int(p); self.delay=int(delay); self.gamma=float(gamma); self.band=float(band)
        self.mid_mode=mid_mode; self.mode=mode
        self.params_={}; self.masks_={}; self.selected_=None

    def _split(self, zc):
        L = zc < (self.gamma - self.band)
        U = zc > (self.gamma + self.band)
        M = ~(L | U); return L, M, U

    def fit(self, y, z):
        y = np.asarray(y, float); z = np.asarray(z, float)
        Y, X = lag_matrix(y, self.p)
        start = self.p - self.delay; end = -self.delay if self.delay>0 else None
        zc = z[start:end]
        if len(zc) != len(Y): raise ValueError("Length mismatch; check p/delay vs. y/z length.")
        L, M, U = self._split(zc); self.masks_ = {"L": L, "M": M, "U": U}
        def reg(mask):
            if mask.sum() < X.shape[1] + 1: return None
            beta, *_ = lstsq(X[mask], Y[mask], rcond=None); return beta
        if self.mode == "two":
            IN, OUT = M, (L | U); self.params_ = {"IN": reg(IN), "OUT": reg(OUT)}
        else:
            self.params_ = {"L": reg(L), "M": reg(M) if self.mid_mode=="separate" else None, "U": reg(U)}
        self.X_, self.Y_, self.zc_ = X, Y, zc; return self

    def predict_one(self, y_hist, z_next, last_regime="M"):
        x = np.r_[1.0, y_hist[-self.p:][::-1]]
        if self.mode == "two":
            inside = abs(z_next - self.gamma) <= self.band
            beta = self.params_["IN"] if inside else self.params_["OUT"]
            if beta is None: beta = self.params_.get("OUT" if inside else "IN")
            if beta is None: beta = np.zeros_like(x)
            return float(np.dot(x, beta)), ("IN" if inside else "OUT")
        else:
            if z_next < (self.gamma - self.band): beta=self.params_["L"]; reg="L"
            elif z_next > (self.gamma + self.band): beta=self.params_["U"]; reg="U"
            else:
                reg="M"; beta=self.params_["M"]
                if beta is None: beta = self.params_.get(last_regime) or np.zeros_like(x)
            return float(np.dot(x, beta)), reg

    @staticmethod
    def grid_search(y, z, p_list=(1,2), delay_list=(1,2), gamma_grid=None, band_grid=None, mid_mode="separate", mode="two"):
        y = np.asarray(y, float); z = np.asarray(z, float)
        zf = z[np.isfinite(z)]
        if zf.size == 0: raise ValueError("z has no finite values.")
        if gamma_grid is None: gamma_grid = np.percentile(zf, [30, 50, 70])
        if band_grid is None:
            sd = np.nanstd(z); 
            if not np.isfinite(sd) or sd == 0: sd = 1.0
            band_grid = np.linspace(sd*0.2, sd*0.8, 3)
        best=None; best_obj=np.inf
        for p in p_list:
            for d in delay_list:
                for g in gamma_grid:
                    for b in band_grid:
                        try:
                            m = BufferedTAR(p=p, delay=d, gamma=float(g), band=float(b), mid_mode=mid_mode, mode=mode).fit(y, z)
                        except Exception:
                            continue
                        Y, X = lag_matrix(y, p)
                        if mode == "two":
                            IN = m.masks_["M"]; OUT = m.masks_["L"] | m.masks_["U"]
                            betas=[m.params_.get("IN"), m.params_.get("OUT")]
                            betas=[np.zeros(X.shape[1]) if bb is None else bb for bb in betas]
                            idx = np.where(OUT,1,0); B=np.vstack(betas).T
                            yhat = np.sum(X * B[:, idx].T, axis=1)
                        else:
                            L,M,U = m.masks_["L"], m.masks_["M"], m.masks_["U"]
                            idx = np.zeros_like(L,int); idx[M]=1; idx[U]=2
                            betas=[m.params_.get("L"), m.params_.get("M"), m.params_.get("U")]
                            betas=[np.zeros(X.shape[1]) if bb is None else bb for bb in betas]
                            B=np.vstack(betas).T; yhat = np.sum(X * B[:, idx].T, axis=1)
                        resid = Y - yhat; sse = float(resid @ resid)
                        if sse < best_obj: best_obj = sse; best = m
        if best is None: raise RuntimeError("Grid search failed.")
        best.selected_ = {"objective": best_obj}; return best
