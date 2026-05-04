# ============================================================
# OUTLIER BASELINES PACK (SCALABLE)
# - RAW coefficient space baselines:
#   (1) Isolation Forest
#   (2) LOF (novelty=True)
#   (3) One-Class SVM (RBF)  [fit on subsample for scalability]
#   (4) RFF+PCA residual (approx. RBF Kernel PCA residual)
#
# Train on TRAIN split only, evaluate on TEST split
# Report: AUROC, AUPRC, Enrichment@tau (tau=0.95,0.99) + counts
# Outputs:
#   1) outlier_baselines_test.csv
#   2) outlier_baselines_test_table.tex
# ============================================================

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, average_precision_score

from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

from sklearn.kernel_approximation import RBFSampler
from sklearn.decomposition import PCA

# ----------------------------
# CONFIG
# ----------------------------
SEED = 42
TAUS = [0.95, 0.99]
TARGETS = [8, 10]
INVARIANTS = ["Alexander", "Jones", "HOMFLY"]

EPS = 1e-12

LOF_K = 35
OCSVM_NU = 0.05

# Scalability knobs
SVM_MAX_TRAIN = 30000      # subsample train for OCSVM (tune 10k-50k)
LOF_MAX_TRAIN = None       # LOF needs full train for novelty; keep None unless too slow
IF_N_EST = 300

# RFF baseline (approx KPCA residual)
RFF_D = 2048               # #random Fourier features (1024-4096)
PCA_D = 16                 # residual in RFF space after linear compression

# ----------------------------
# Preconditions
# ----------------------------
required = ["sig_f", "train_idx", "test_idx", "X_A_f", "X_J_f", "X_H_f"]
missing = [v for v in required if v not in globals()]
if missing:
    raise RuntimeError(f"Missing required variables in runtime: {missing}")

y_all = np.asarray(sig_f).astype(int)
train_idx = np.asarray(train_idx)
test_idx  = np.asarray(test_idx)

X_raw_store = {
    "Alexander": np.asarray(X_A_f, dtype=np.float32),
    "Jones":     np.asarray(X_J_f, dtype=np.float32),
    "HOMFLY":    np.asarray(X_H_f, dtype=np.float32),
}

# ----------------------------
# Metrics helpers
# ----------------------------
def safe_auroc(y_true, scores):
    y_true = np.asarray(y_true).astype(int)
    if len(np.unique(y_true)) < 2:
        return np.nan
    return roc_auc_score(y_true, scores)

def safe_auprc(y_true, scores):
    y_true = np.asarray(y_true).astype(int)
    if y_true.sum() == 0:
        return np.nan
    return average_precision_score(y_true, scores)

def topk_tail_mask(scores, tau):
    """
    Exact fixed-mass top-k tail.

    Selects exactly ceil((1 - tau) * n) samples with the largest scores.
    Ties are broken deterministically by stable sorting.
    """
    scores = np.asarray(scores, dtype=np.float64).ravel()
    n = int(len(scores))
    k = int(np.ceil((1.0 - tau) * n))

    if k <= 0:
        raise ValueError(f"tau={tau} gives k={k}.")
    if k > n:
        raise ValueError(f"tau={tau} gives k={k} > n={n}.")

    order = np.argsort(-scores, kind="mergesort")
    mask = np.zeros(n, dtype=bool)
    mask[order[:k]] = True
    return mask

def enrichment_at_tau_with_counts(scores, y_true, tau):
    """
    Exact fixed-mass top-k enrichment.

    Uses exactly ceil((1 - tau) * n) samples with the largest scores.
    """
    scores = np.asarray(scores, dtype=np.float64).ravel()
    y_true = np.asarray(y_true).astype(int).ravel()

    if len(scores) != len(y_true):
        raise ValueError(
            f"Length mismatch: len(scores)={len(scores)}, len(y_true)={len(y_true)}"
        )

    n = int(len(y_true))
    base_rate = float(y_true.mean()) if n > 0 else np.nan
    n_pos = int(y_true.sum())

    is_tail = topk_tail_mask(scores, tau)
    tail_size = int(is_tail.sum())
    n_pos_tail = int(y_true[is_tail].sum())

    if base_rate == 0 or tail_size == 0:
        return np.nan, n_pos, n_pos_tail, tail_size

    tail_rate = n_pos_tail / tail_size
    enrichment = tail_rate / base_rate

    return float(enrichment), n_pos, n_pos_tail, tail_size

def rows_for_method(scores_test, y_sig_test, method_name, inv_name, taus=TAUS, targets=TARGETS):
    out = []
    for s in targets:
        ybin = (np.abs(y_sig_test) >= s).astype(int)
        auroc = safe_auroc(ybin, scores_test)
        auprc = safe_auprc(ybin, scores_test)
        for tau in taus:
            enr, n_pos, n_pos_tail, tail_size = enrichment_at_tau_with_counts(scores_test, ybin, tau)
            out.append({
                "Invariant": inv_name,
                "Method": method_name,
                "Target": f"Y{s}",
                "Tau": float(tau),
                "AUROC": float(auroc) if np.isfinite(auroc) else np.nan,
                "AUPRC": float(auprc) if np.isfinite(auprc) else np.nan,
                "Enrichment": float(enr) if np.isfinite(enr) else np.nan,
                "Positives": int(n_pos),
                "Tail_captured": int(n_pos_tail),
                "Tail_size": int(tail_size),
                "N_test": int(len(ybin)),
                "Base_rate": float(ybin.mean()),
            })
    return out

rng = np.random.default_rng(SEED)

# ----------------------------
# Baseline scorers
# ----------------------------
def fit_score_isoforest(Xtr, Xte):
    pipe = Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("iso", IsolationForest(
            n_estimators=IF_N_EST,
            contamination="auto",
            random_state=SEED,
            n_jobs=-1
        ))
    ])
    pipe.fit(Xtr)
    Xte_s = pipe.named_steps["scaler"].transform(Xte)
    scores = -pipe.named_steps["iso"].decision_function(Xte_s)  # higher = more outlier
    return scores

def fit_score_lof(Xtr, Xte, k=LOF_K):
    # LOF novelty needs to store neighborhood structure; can be heavy but not NxN.
    # If too slow, reduce LOF_MAX_TRAIN by subsampling (but then it's no longer "fit on all train").
    pipe = Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("lof", LocalOutlierFactor(
            n_neighbors=k,
            novelty=True,
            metric="minkowski",
            n_jobs=-1
        ))
    ])
    pipe.fit(Xtr)
    Xte_s = pipe.named_steps["scaler"].transform(Xte)
    scores = -pipe.named_steps["lof"].decision_function(Xte_s)  # higher = more outlier
    return scores

def fit_score_ocsvm_subsample(Xtr, Xte, nu=OCSVM_NU, max_train=SVM_MAX_TRAIN):
    # OCSVM is quadratic-ish; subsample for feasibility.
    n = len(Xtr)
    if (max_train is not None) and (n > max_train):
        idx = rng.choice(n, size=max_train, replace=False)
        Xfit = Xtr[idx]
        fit_note = f"sub={max_train}"
    else:
        Xfit = Xtr
        fit_note = f"full={n}"

    pipe = Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("svm", OneClassSVM(kernel="rbf", nu=nu, gamma="scale"))
    ])
    pipe.fit(Xfit)
    Xte_s = pipe.named_steps["scaler"].transform(Xte)
    scores = -pipe.named_steps["svm"].decision_function(Xte_s).ravel()
    return scores, fit_note

def fit_score_rff_pca_residual(Xtr, Xte, rff_d=RFF_D, pca_d=PCA_D):
    # Approximate RBF kernel map phi(x) via Random Fourier Features:
    # phi(x) in R^rff_d, then do PCA residual in that space.
    scaler = StandardScaler(with_mean=True, with_std=True)
    Xtr_s = scaler.fit_transform(Xtr)
    Xte_s = scaler.transform(Xte)

    # gamma="scale" ~ 1/(n_features * var). After standardization var~1 => gamma ~ 1/n_features
    gamma = 1.0 / max(Xtr_s.shape[1], 1)

    rff = RBFSampler(gamma=gamma, n_components=rff_d, random_state=SEED)
    Phi_tr = rff.fit_transform(Xtr_s)  # (n_train, rff_d) - dense but manageable
    Phi_te = rff.transform(Xte_s)

    # PCA in feature space, fit on train only
    pca = PCA(n_components=pca_d, random_state=SEED)
    Z_tr = pca.fit_transform(Phi_tr)
    Z_te = pca.transform(Phi_te)
    Phi_hat_te = pca.inverse_transform(Z_te)

    # residual in RFF space + normalize by ||phi||^2
    resid = np.sum((Phi_te - Phi_hat_te) ** 2, axis=1)
    den = np.sum(Phi_te ** 2, axis=1) + EPS
    scores = resid / den  # higher = more outlier under kernel geometry
    return scores, f"rff={rff_d},d={pca_d}"

# ============================================================
# MAIN: Run baselines (TRAIN->TEST) for each invariant
# ============================================================
all_rows = []
y_test_sig = y_all[test_idx]

for inv in INVARIANTS:
    X = X_raw_store[inv]
    Xtr = X[train_idx]
    Xte = X[test_idx]

    print(f"\n=== {inv}: fitting baselines on TRAIN, scoring TEST ===")
    print("Shapes:", Xtr.shape, Xte.shape)

    # Isolation Forest
    sc_iso = fit_score_isoforest(Xtr, Xte)
    all_rows += rows_for_method(sc_iso, y_test_sig, "IsolationForest(raw)", inv)

    # LOF
    sc_lof = fit_score_lof(Xtr, Xte, k=LOF_K)
    all_rows += rows_for_method(sc_lof, y_test_sig, f"LOF(raw,k={LOF_K})", inv)

    # One-Class SVM (subsample)
    sc_svm, note = fit_score_ocsvm_subsample(Xtr, Xte, nu=OCSVM_NU, max_train=SVM_MAX_TRAIN)
    all_rows += rows_for_method(sc_svm, y_test_sig, f"OCSVM-RBF(raw,nu={OCSVM_NU},{note})", inv)

    # Approx Kernel PCA residual via RFF+PCA residual
    sc_rff, note = fit_score_rff_pca_residual(Xtr, Xte, rff_d=RFF_D, pca_d=PCA_D)
    all_rows += rows_for_method(sc_rff, y_test_sig, f"RFF+PCA-resid(raw,{note})", inv)

df_base = pd.DataFrame(all_rows)
df_base = df_base.sort_values(["Invariant","Target","Tau","Method"]).reset_index(drop=True)
df_base.to_csv("outlier_baselines_test.csv", index=False)
print("\nSaved: outlier_baselines_test.csv")
print(df_base.head(30))

# ============================================================
# Build a compact LaTeX table for JONES (main paper)
# ============================================================
def fmt_x(x, nd=3):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "--"
    return f"{x:.{nd}f}"

def fmt_enr(enr, cap, tail):
    if enr is None or (isinstance(enr, float) and not np.isfinite(enr)):
        return "--"
    return f"{enr:.2f}$\\times$ ({cap}/{tail})"

inv_main = "Jones"
dfj = df_base[df_base["Invariant"] == inv_main].copy()

table_rows = []
for method in dfj["Method"].unique():
    for tgt in [f"Y{s}" for s in TARGETS]:
        sub = dfj[(dfj["Method"] == method) & (dfj["Target"] == tgt)]
        auroc = sub["AUROC"].iloc[0] if len(sub) else np.nan
        auprc = sub["AUPRC"].iloc[0] if len(sub) else np.nan

        sub95 = sub[np.isclose(sub["Tau"], 0.95)]
        sub99 = sub[np.isclose(sub["Tau"], 0.99)]

        enr95 = sub95["Enrichment"].iloc[0] if len(sub95) else np.nan
        cap95 = int(sub95["Tail_captured"].iloc[0]) if len(sub95) else 0
        tail95 = int(sub95["Tail_size"].iloc[0]) if len(sub95) else 0

        enr99 = sub99["Enrichment"].iloc[0] if len(sub99) else np.nan
        cap99 = int(sub99["Tail_captured"].iloc[0]) if len(sub99) else 0
        tail99 = int(sub99["Tail_size"].iloc[0]) if len(sub99) else 0

        table_rows.append((method, tgt, auroc, auprc, (enr95, cap95, tail95), (enr99, cap99, tail99)))

tex_lines = []
tex_lines.append(r"\begin{table}[t]")
tex_lines.append(r"\centering")
tex_lines.append(r"\caption{\textbf{Outlier-detection baselines on raw coefficients (Jones).} "
                 r"Models are fit on the training set and evaluated on the test set. "
                 r"We report AUROC, AUPRC, and enrichment at $\tau\in\{0.95,0.99\}$ "
                 r"(counts shown as positives captured / tail size). "
                 r"\emph{Kernel PCA residual is approximated via random Fourier features (RFF).}}")
tex_lines.append(r"\label{tab:outlier_baselines_raw}")
tex_lines.append(r"\begin{small}")
tex_lines.append(r"\begin{tabular}{lcccc}")
tex_lines.append(r"\toprule")
tex_lines.append(r"Method & Target & AUROC & AUPRC & Enr.@0.95 / Enr.@0.99 \\")
tex_lines.append(r"\midrule")

for method, tgt, auroc, auprc, e95, e99 in table_rows:
    tex_lines.append(
        f"{method} & {tgt} & {fmt_x(auroc)} & {fmt_x(auprc)} & {fmt_enr(*e95)} / {fmt_enr(*e99)} \\\\"
    )

tex_lines.append(r"\bottomrule")
tex_lines.append(r"\end{tabular}")
tex_lines.append(r"\end{small}")
tex_lines.append(r"\end{table}")

tex = "\n".join(tex_lines)
with open("outlier_baselines_test_table.tex", "w") as f:
    f.write(tex)

print("\nSaved: outlier_baselines_test_table.tex")
print("\n--- LaTeX TABLE PREVIEW ---\n")
print(tex)
