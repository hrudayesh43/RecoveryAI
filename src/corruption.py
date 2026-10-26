import numpy as np
import pandas as pd

def corrupt_dataframe(df: pd.DataFrame, scenario: str, intensity: float, seed: int = 42) -> pd.DataFrame:
    """Create controlled test corruption. Never modifies the caller's dataframe."""
    out = df.copy(deep=True)
    if out.empty or out.shape[1] == 0:
        return out
    rng = np.random.default_rng(seed)
    intensity = min(max(float(intensity), 0.0), 1.0)
    n = max(1, int(round(len(out) * intensity))) if len(out) else 0
    numeric = list(out.select_dtypes(include="number").columns)
    candidates = [c for c in out.columns if out[c].notna().any()]
    if scenario == "Missing values":
        for col in candidates[:max(1, min(2, len(candidates)))]:
            idx = rng.choice(out.index.to_numpy(), size=min(n, len(out)), replace=False)
            out.loc[idx, col] = np.nan
    elif scenario == "Duplicate rows" and len(out) > 0:
        sample = out.sample(n=min(n, len(out)), replace=True, random_state=seed)
        out = pd.concat([out, sample], ignore_index=True)
    elif scenario == "Numeric outliers" and numeric:
        col = numeric[0]
        idx = rng.choice(out.index.to_numpy(), size=min(n, len(out)), replace=False)
        vals = pd.to_numeric(out.loc[idx, col], errors="coerce")
        scale = float(pd.to_numeric(out[col], errors="coerce").std())
        if not np.isfinite(scale) or scale == 0:
            scale = 1.0
        out.loc[idx, col] = vals + (scale * 8)
    elif scenario == "Feature drift" and numeric:
        col = numeric[0]
        values = pd.to_numeric(out[col], errors="coerce")
        scale = float(values.std())
        if not np.isfinite(scale) or scale == 0:
            scale = 1.0
        idx = rng.choice(out.index.to_numpy(), size=min(n, len(out)), replace=False)
        out.loc[idx, col] = pd.to_numeric(out.loc[idx, col], errors="coerce") + 2 * scale
    return out
