#!/usr/bin/env python3
# coding: utf-8
"""
Collect per-file commits between Intro Hash and Payment Hash untuk SELF-FIXED issues,
dan bandingkan kontribusi:
- Self-fixer (introducer == fixer)
- Others (developer lain)

Output:
- CSV per file dengan count & ratio commit per role
- R4.8: per-file change-size share berbasis code churn (added + deleted lines)
- Boxplot COUNT & RATIO (INTRO+MID+PAYMENT) -> Self-fixer vs Others
- Boxplot RATIO MID-only                   -> Self-fixer vs Others
- Statistical tests (MOST APPROPRIATE):
    * One-sample Wilcoxon signed-rank test on self_share vs 0.5 (one-sided, greater)
    * Effect size: matched-pairs rank-biserial correlation (RBC)
    * Dominance rate: Pr(self_share > 0.5)
    * Sign test (binomial test) as robust alternative
Notes:
- Do NOT run two-sample tests Self vs Others for ratios, because Others = 1 - Self by construction.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional
from functools import lru_cache
from git import Repo as GitPythonRepo



# ====== BIGGER FONTS (GLOBAL) ======
plt.rcParams.update({
    "font.size": 14,          # base
    "axes.labelsize": 12,     # y-label
    "axes.titlesize": 12,     # title (if you use it)
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
})

# --- Stats imports (SciPy) ---
try:
    from scipy.stats import wilcoxon, rankdata, binomtest, spearmanr
except Exception as e:
    raise SystemExit(
        "This script requires SciPy for Wilcoxon/rankdata/binomtest.\n"
        "Install with: pip install scipy\n"
        f"Original import error: {e}"
    )

# ====== CONFIG: ubah jika perlu ======
# COMMITS_CSV = "/media/edsu/Drive-D2/Python/ICSA-ATD-Lifecycle-To-Replicate/rq2/SF/SELF-FIXED_commits_between_periods_autorepo-ONLY-INTRO-ALL-AUTHORS-SELF-FIXED_UPGRADED-ATD-FINAL-DATASET-1091-TRACED-ALL-EXTS.csv" #ALL DATA TRUE AND WEAK ATD

# Current location of local Git repositories
REPOSITORIES_DIRECTORY = "/media/edsu/Drive-D2/Python/ICSA-ATD-Lifecycle-To-Replicate/ATD-DATA/repos"

# TRUE ATD ONLY
COMMITS_CSV = "/media/edsu/Drive-D2/Python/ICSA-ATD-Lifecycle-To-Replicate/rq2/SF/SELF-FIXED_commits_between_periods_ATD-NO-WEAK.csv" # TRUE ATD ONLY

OUT_CSV           = "/media/edsu/Drive-D2/Python/ICSA-ATD-Lifecycle-To-Replicate/rq2/SF/TRUE-ATD-ONLY/rq2_self_fixer_vs_others_per_file_WITH_ENDPOINTS.csv"  #OUTPUT
OUT_PDF_CNT       = "/media/edsu/Drive-D2/Python/ICSA-ATD-Lifecycle-To-Replicate/rq2/SF/TRUE-ATD-ONLY/rq2_box_counts_per_file_SELF-FIXER_vs_OTHERS_WITH_ENDPOINTS.pdf"
OUT_PDF_RATIO     = "/media/edsu/Drive-D2/Python/ICSA-ATD-Lifecycle-To-Replicate/rq2/SF/TRUE-ATD-ONLY/rq2_box_RATIO_per_file_SELF-FIXER_vs_OTHERS_WITH_ENDPOINTS.pdf"
OUT_PDF_RATIO_MID = "/media/edsu/Drive-D2/Python/ICSA-ATD-Lifecycle-To-Replicate/rq2/SF/TRUE-ATD-ONLY/rq2_box_RATIO_per_file_SELF-FIXER_vs_OTHERS_MID_ONLY.pdf"

# NEW: stats output (CSV + TXT)
OUT_STATS_CSV     = "/media/edsu/Drive-D2/Python/ICSA-ATD-Lifecycle-To-Replicate/rq2/SF/TRUE-ATD-ONLY/rq2_self_fixer_vs_0p5_stats.csv"
OUT_STATS_TXT     = "/media/edsu/Drive-D2/Python/ICSA-ATD-Lifecycle-To-Replicate/rq2/SF/TRUE-ATD-ONLY/rq2_self_fixer_vs_0p5_stats.txt"

# R4.8: change-size (code churn = added + deleted lines) outputs
OUT_PDF_CHURN_RATIO = "/media/edsu/Drive-D2/Python/ICSA-ATD-Lifecycle-To-Replicate/rq2/SF/TRUE-ATD-ONLY/rq2_box_CHANGE_SIZE_RATIO_per_file_SELF-FIXER_vs_OTHERS_WITH_ENDPOINTS.pdf"
OUT_PDF_CHURN_RATIO_MID = "/media/edsu/Drive-D2/Python/ICSA-ATD-Lifecycle-To-Replicate/rq2/SF/TRUE-ATD-ONLY/rq2_box_CHANGE_SIZE_RATIO_per_file_SELF-FIXER_vs_OTHERS_MID_ONLY.pdf"
OUT_STATS_R48_CSV = "/media/edsu/Drive-D2/Python/ICSA-ATD-Lifecycle-To-Replicate/rq2/SF/TRUE-ATD-ONLY/rq2_R4_8_commit_vs_change_size_stats.csv"
OUT_STATS_R48_TXT = "/media/edsu/Drive-D2/Python/ICSA-ATD-Lifecycle-To-Replicate/rq2/SF/TRUE-ATD-ONLY/rq2_R4_8_commit_vs_change_size_stats.txt"
OUT_CHURN_DIAGNOSTICS = "/media/edsu/Drive-D2/Python/ICSA-ATD-Lifecycle-To-Replicate/rq2/SF/TRUE-ATD-ONLY/rq2_R4_8_churn_unmatched_rows.csv"

# >>> Hanya gunakan pasangan Key/File yang memiliki MID?
FILTER_REQUIRE_MID = True

# >>> FILTER ekstensi file (True/False) <<<
FILTER_BY_EXT = False
ALLOWED_EXTS = {
    ".java", ".scala", ".kt", ".kts", ".py", ".rb", ".go", ".c", ".cc", ".cpp",
    ".h", ".hpp", ".cs", ".php", ".ts", ".tsx", ".jsx", ".rs", ".swift",
    ".erl", ".ex", ".exs", ".xml"
}
# =======================================

# --- GLOBAL FONT/THEME ---
plt.rcParams.update({
    "figure.dpi": 180,
    "savefig.dpi": 180,
    "font.size": 14,
    "axes.titlesize": 18,
    "axes.labelsize": 15,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
})

# ==== THEME: dua warna konsisten ====
COL_SELF  = "#01BFC4"   # teal  (Self-fixer)
COL_OTHER = "#999999"   # grey  (Others)

def boxplot_multi_series(ax, data_list, tick_labels, y_label, title, ylim=None):
    bp = ax.boxplot(
        data_list,
        tick_labels=tick_labels,
        showmeans=True,
        patch_artist=True
    )
    fills = [COL_SELF, COL_OTHER]

    for i, box in enumerate(bp['boxes']):
        box.set(facecolor=fills[i % len(fills)], edgecolor="black", alpha=0.9, linewidth=1.4)
    for w in bp['whiskers']:
        w.set(color="black", linewidth=1.4)
    for c in bp['caps']:
        c.set(color="black", linewidth=1.4)
    for med in bp['medians']:
        med.set(color="black", linewidth=2.4)
    for mean in bp['means']:
        mean.set(marker="D", markerfacecolor="white", markeredgecolor="black", markersize=8)
    for i, fl in enumerate(bp['fliers']):
        fl.set(marker='o', alpha=0.55, markersize=4.5,
               markerfacecolor=fills[i % len(fills)], markeredgecolor="black")

    ax.set_ylabel(y_label, fontsize=15)
    ax.set_title(title, fontsize=18)
    ax.tick_params(axis='both', labelsize=13)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(axis='y', linestyle=':', linewidth=0.8, alpha=0.6)

def norm_str(x: Optional[str]) -> str:
    return ("" if x is None else str(x)).strip().lower()

def norm_role(r):
    rr = norm_str(r)
    if rr in {"intro", "inro", "introduction"}: return "INTRO"
    if rr in {"payment", "pay"}:                return "PAYMENT"
    if rr in {"mid", "between"}:                return "MID"
    return str(r)

def get_ext(path: str) -> str:
    return os.path.splitext(path)[1].lower()

def author_id(row):
    for c in ["Author_Email", "Author_Name", "Committer_Email", "Committer_Name"]:
        if c in row and str(row[c]).strip():
            return norm_str(row[c])
    return ""


# ============================================================
# R4.8 HELPERS: per-commit, per-file change size (code churn)
# ============================================================

def as_bool(x) -> bool:
    """Robust conversion of CSV truth values."""
    return norm_str(x) in {"1", "true", "t", "yes", "y"}


def norm_path(path: str) -> str:
    return str(path or "").strip().replace("\\", "/")


# Old repository directory names in the CSV -> current directory candidates.
# Some projects currently exist in both lower- and upper-case directories
# (e.g., camel/CAMEL and spark/SPARK). The resolver below verifies the
# Commit_Hash, so it can choose the directory that actually contains the commit.
REPO_FOLDER_CANDIDATES = {
    "AMQ-activemq": ["activemq"],
    "AMQ-activemq-cpp": ["activemq-cpp"],
    "CAMEL": ["camel", "CAMEL"],
    "CAMEL-camel-spring-boot": ["camel-spring-boot"],
    "CAMEL-camel-k": ["camel-k"],
    "CAMEL-camel-quarkus": ["camel-quarkus"],
    "CASSANDRA": ["cassandra"],
    "CASSANDRA-cassandra-analytics": ["cassandra-analytics"],
    "DRILL": ["drill"],
    "GEODE": ["geode"],
    "KAFKA": ["kafka"],
    "LUCENE": ["lucene"],
    "NETBEANS": ["netbeans"],
    "SOLR": ["solr"],
    "SOLR-solr-sandbox": ["solr-sandbox"],
    "SPARK": ["spark", "SPARK"],
}


@lru_cache(maxsize=50000)
def repo_contains_commit(repo_path: str, commit_hash: str) -> bool:
    """Return True if repo_path is a Git repository containing commit_hash."""
    if not repo_path or not os.path.isdir(repo_path):
        return False
    if not commit_hash:
        return True
    try:
        repo = GitPythonRepo(repo_path)
        repo.commit(str(commit_hash).strip())
        return True
    except Exception:
        return False


def _existing_repo_dirs():
    """List current repository directories under REPOSITORIES_DIRECTORY."""
    if not os.path.isdir(REPOSITORIES_DIRECTORY):
        return []
    return [
        name for name in os.listdir(REPOSITORIES_DIRECTORY)
        if os.path.isdir(os.path.join(REPOSITORIES_DIRECTORY, name))
    ]


def resolve_repo_path(csv_repo_path: str, commit_hash: str = "", csv_repo_name: str = "") -> str:
    """
    Resolve a CSV repository path to the CURRENT repository directory.

    Resolution strategy:
      1. Use the CSV path if it still exists and contains Commit_Hash.
      2. Try the explicit old-name -> current-folder mapping above.
      3. Try exact/case-insensitive folder-name variants.
      4. As a last resort, search current repositories for Commit_Hash.

    Commit verification is important because Linux is case-sensitive and the
    current repository root contains both camel/CAMEL and spark/SPARK.
    """
    raw = str(csv_repo_path or "").strip()
    commit_hash = str(commit_hash or "").strip()
    csv_repo_name = str(csv_repo_name or "").strip()

    # 1) Old CSV path may still be valid.
    if raw and repo_contains_commit(raw, commit_hash):
        return raw

    old_folder = os.path.basename(os.path.normpath(raw)) if raw else ""

    candidate_names = []

    # 2) Explicit mapping from old folder names to current names.
    if old_folder in REPO_FOLDER_CANDIDATES:
        candidate_names.extend(REPO_FOLDER_CANDIDATES[old_folder])

    # Repo_Name can also help if present in the CSV.
    if csv_repo_name:
        candidate_names.extend([csv_repo_name, csv_repo_name.lower(), csv_repo_name.upper()])

    # 3) Exact and case variants of the old basename.
    if old_folder:
        candidate_names.extend([old_folder, old_folder.lower(), old_folder.upper()])

        # Old names sometimes carried a project prefix, e.g.
        # CASSANDRA-cassandra-analytics -> cassandra-analytics.
        if "-" in old_folder:
            candidate_names.append(old_folder.split("-", 1)[1])

    # De-duplicate while preserving order.
    candidate_names = list(dict.fromkeys(x for x in candidate_names if x))

    # First prefer candidates that actually contain this commit.
    for name in candidate_names:
        candidate = os.path.join(REPOSITORIES_DIRECTORY, name)
        if repo_contains_commit(candidate, commit_hash):
            return candidate

    # Case-insensitive name matching against folders that actually exist.
    current_dirs = _existing_repo_dirs()
    lowered = {name.lower(): [] for name in current_dirs}
    for name in current_dirs:
        lowered.setdefault(name.lower(), []).append(name)

    for wanted in candidate_names:
        matches = lowered.get(wanted.lower(), [])
        for actual_name in matches:
            candidate = os.path.join(REPOSITORIES_DIRECTORY, actual_name)
            if repo_contains_commit(candidate, commit_hash):
                return candidate

    # 4) Last resort: locate the commit itself among current repositories.
    # This also handles future folder renames without another code change.
    if commit_hash:
        for actual_name in current_dirs:
            candidate = os.path.join(REPOSITORIES_DIRECTORY, actual_name)
            if repo_contains_commit(candidate, commit_hash):
                return candidate

    # Nothing matched. Return the first intended candidate for diagnostics.
    if candidate_names:
        return os.path.join(REPOSITORIES_DIRECTORY, candidate_names[0])

    return ""


@lru_cache(maxsize=64)
def get_repo(repo_path: str):
    """Cache GitPython Repo objects."""
    return GitPythonRepo(repo_path)


@lru_cache(maxsize=20000)
def get_commit_numstat(repo_path: str, commit_hash: str):
    """
    Return per-file change statistics for one commit.

    The metric used for R4.8 is code churn:
        changed_lines = insertions + deletions

    GitPython obtains these values from Git's numstat information.
    The result is cached because the same commit can occur for multiple
    (Key, File) rows.
    """
    try:
        repo = get_repo(repo_path)
        commit = repo.commit(commit_hash)
        files = commit.stats.files or {}
        result = {}
        for p, st in files.items():
            pp = norm_path(p)
            added = int(st.get("insertions", 0) or 0)
            deleted = int(st.get("deletions", 0) or 0)
            result[pp] = {
                "added": added,
                "deleted": deleted,
                "changed": added + deleted,
            }
        return result
    except Exception as exc:
        return {"__ERROR__": str(exc)}


def resolve_numstat_for_target(repo_path: str, commit_hash: str, target_file: str):
    """
    Resolve the changed-line count for the tracked file in a commit.

    Matching order:
      1) exact normalized path;
      2) unique suffix match (helps with simple path-prefix differences);
      3) unique basename match (last-resort diagnostic fallback).

    Returns: (added, deleted, changed, match_mode, error)

    IMPORTANT: rows that cannot be mapped unambiguously are marked unmatched;
    they are NOT silently assigned zero churn.
    """
    target = norm_path(target_file)
    stats = get_commit_numstat(str(repo_path), str(commit_hash))

    if "__ERROR__" in stats:
        return np.nan, np.nan, np.nan, "git_error", stats["__ERROR__"]

    if target in stats:
        st = stats[target]
        return st["added"], st["deleted"], st["changed"], "exact", ""

    # Unique suffix match
    suffix_matches = [
        p for p in stats
        if p.endswith("/" + target) or target.endswith("/" + p)
    ]
    if len(suffix_matches) == 1:
        st = stats[suffix_matches[0]]
        return st["added"], st["deleted"], st["changed"], "suffix", ""

    # Unique basename match (only if unambiguous within this commit)
    base = os.path.basename(target)
    basename_matches = [p for p in stats if os.path.basename(p) == base]
    if len(basename_matches) == 1:
        st = stats[basename_matches[0]]
        return st["added"], st["deleted"], st["changed"], "basename", ""

    return np.nan, np.nan, np.nan, "unmatched", "file not found unambiguously in commit numstat"


def attach_change_size(df_contrib: pd.DataFrame) -> pd.DataFrame:
    """
    Add Added_Lines, Deleted_Lines, Changed_Lines to contribution rows.
    Only rows that actually touch the tracked file should be passed here.
    """
    records = []
    total = len(df_contrib)
    for i, (_, row) in enumerate(df_contrib.iterrows(), start=1):
        repo_path_original = row.get("Repo_Path", "")
        repo_name_original = row.get("Repo_Name", "")
        commit_hash = row.get("Commit_Hash", "")
        file_path = row.get("File_Path_std", "")
        repo_path = resolve_repo_path(
            repo_path_original,
            commit_hash=commit_hash,
            csv_repo_name=repo_name_original,
        )

        if not repo_path or not commit_hash or not file_path:
            added = deleted = changed = np.nan
            mode = "missing_input"
            error = "missing Repo_Path, Commit_Hash, or File_Path"
        elif not os.path.exists(repo_path):
            added = deleted = changed = np.nan
            mode = "repo_missing"
            error = f"repository path not found: {repo_path}"
        else:
            added, deleted, changed, mode, error = resolve_numstat_for_target(
                repo_path, commit_hash, file_path
            )

        rec = row.to_dict()
        rec["Repo_Path_Original"] = repo_path_original
        rec["Repo_Name_Original"] = repo_name_original
        rec["Repo_Path_Resolved"] = repo_path
        rec["Added_Lines"] = added
        rec["Deleted_Lines"] = deleted
        rec["Changed_Lines"] = changed
        rec["Churn_Match_Mode"] = mode
        rec["Churn_Error"] = error
        records.append(rec)

        if i % 5000 == 0 or i == total:
            print(f"R4.8 numstat: processed {i}/{total} contribution rows")

    return pd.DataFrame(records)

# =========================
# Statistical test helpers
# =========================
def _safe_float_array(x) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return x[~np.isnan(x)]

def one_sample_wilcoxon_vs_mu0(x: np.ndarray, mu0: float = 0.5, alternative: str = "greater"):
    """
    One-sample Wilcoxon signed-rank test on d = x - mu0.
    Returns dict: n_total, n_used, W, p, RBC, median, q25, q75, pr_gt_mu0, sign_p
    """
    x = _safe_float_array(x)
    n_total = int(x.size)

    if n_total == 0:
        return {
            "n_total": 0, "n_used": 0, "W": np.nan, "p": np.nan, "RBC": np.nan,
            "median": np.nan, "q25": np.nan, "q75": np.nan, "pr_gt_mu0": np.nan, "sign_p": np.nan
        }

    d = x - mu0

    # Wilcoxon discards exact zero differences; do the same for consistent n_used
    d_nz = d[d != 0.0]
    n_used = int(d_nz.size)

    if n_used == 0:
        # All equal to mu0
        pr_gt = float(np.mean(x > mu0))
        return {
            "n_total": n_total, "n_used": 0, "W": 0.0, "p": 1.0, "RBC": 0.0,
            "median": float(np.median(x)),
            "q25": float(np.quantile(x, 0.25)),
            "q75": float(np.quantile(x, 0.75)),
            "pr_gt_mu0": pr_gt,
            "sign_p": 1.0
        }

    res = wilcoxon(d_nz, alternative=alternative, zero_method="wilcox")
    W = float(res.statistic)
    p = float(res.pvalue)

    # Effect size: matched-pairs rank-biserial correlation (RBC)
    ranks = rankdata(np.abs(d_nz), method="average")
    W_pos = float(ranks[d_nz > 0].sum())
    W_neg = float(ranks[d_nz < 0].sum())
    RBC = (W_pos - W_neg) / (W_pos + W_neg)

    # Dominance rate (intuitive)
    pr_gt = float(np.mean(x > mu0))

    # Sign test (binomial): count(d > 0) among non-zero diffs
    n_pos = int(np.sum(d_nz > 0))
    sign_p = float(binomtest(n_pos, n_used, p=0.5, alternative=alternative).pvalue)

    return {
        "n_total": n_total,
        "n_used": n_used,
        "W": W,
        "p": p,
        "RBC": float(RBC),
        "median": float(np.median(x)),
        "q25": float(np.quantile(x, 0.25)),
        "q75": float(np.quantile(x, 0.75)),
        "pr_gt_mu0": pr_gt,
        "sign_p": sign_p
    }

def format_p(p: float) -> str:
    # Avoid printing p=0 due to underflow; show a conventional bound
    if np.isnan(p):
        return "nan"
    if p == 0.0:
        return "< 2.2e-16"
    if p < 1e-4:
        return f"{p:.2e}"
    return f"{p:.6f}"

# ----- Repository root sanity check -----
if not os.path.isdir(REPOSITORIES_DIRECTORY):
    raise SystemExit(f"Repository root not found: {REPOSITORIES_DIRECTORY}")
print(f"Repository root: {REPOSITORIES_DIRECTORY}")

# ----- Load -----
df = pd.read_csv(COMMITS_CSV, dtype=str).fillna("")

if "Role" not in df.columns:
    raise SystemExit("Column 'Role' is required in commits CSV.")
df["Role_std"] = df["Role"].apply(norm_role)

# ----- Kolom File_Path yang robust -----
if "File_Path" in df.columns:
    df["File_Path_std"] = df["File_Path"].astype(str)
elif "Tracked_File" in df.columns:
    df["File_Path_std"] = df["Tracked_File"].astype(str)
elif "Touched_Files" in df.columns:
    df["File_Path_std"] = df["Touched_Files"].apply(
        lambda s: (s.split(";")[0] if isinstance(s, str) and s else "")
    )
else:
    raise SystemExit("Tidak menemukan kolom path file (File_Path/Tracked_File/Touched_Files).")

# ----- FILTER berdasarkan ekstensi file (opsional) -----
if FILTER_BY_EXT:
    df["File_Ext"] = df["File_Path_std"].astype(str).apply(get_ext)
    before_ext = len(df)
    df = df[df["File_Ext"].isin(ALLOWED_EXTS)].copy()
    print(f"Extension filter ON: kept {len(df)} / {before_ext} rows for extensions {sorted(ALLOWED_EXTS)}")
else:
    print("Extension filter OFF: all file types are included.")

# ----- Identitas author -----
df["author_id"] = df.apply(author_id, axis=1)

# ----- Pastikan kolom hash ada untuk dedup per file -----
commit_col = "Commit_Hash" if "Commit_Hash" in df.columns else None
if commit_col is None:
    if "Authored_Datetime" in df.columns and "Commit_Summary" in df.columns:
        df["Commit_Hash"] = df["Authored_Datetime"].astype(str) + "||" + df["Commit_Summary"].astype(str)
    else:
        df["Commit_Hash"] = df.reset_index().index.astype(str)
    commit_col = "Commit_Hash"

# ====== Ambil author INTRO & PAYMENT per (Key, File_Path) ======
intro_auth = (
    df[df["Role_std"] == "INTRO"]
    .groupby(["Key", "File_Path_std"], as_index=False)["author_id"]
    .apply(lambda s: next((x for x in s if x), ""))
    .rename(columns={"author_id": "intro_author"})
)

pay_auth = (
    df[df["Role_std"] == "PAYMENT"]
    .groupby(["Key", "File_Path_std"], as_index=False)["author_id"]
    .apply(lambda s: next((x for x in s if x), ""))
    .rename(columns={"author_id": "pay_author"})
)

authors = pd.merge(intro_auth, pay_auth, on=["Key", "File_Path_std"], how="outer")

for col in ["intro_author", "pay_author"]:
    if col not in authors.columns:
        authors[col] = ""

authors["intro_author_norm"] = authors["intro_author"].apply(norm_str)
authors["pay_author_norm"]   = authors["pay_author"].apply(norm_str)

authors["self_author"] = authors["intro_author_norm"]
mask = (authors["self_author"] == "") & (authors["pay_author_norm"] != "")
authors.loc[mask, "self_author"] = authors.loc[mask, "pay_author_norm"]

# ============================================================
# R4.8: keep only REAL per-file contributions
# ============================================================
# The original CSV may contain endpoint rows retained for lifecycle bookkeeping
# even when the endpoint commit does not touch the tracked file. Such rows are
# useful for identifying the self-fixer, but must NOT contribute to per-file
# commit share or change-size share.
if "Touches_File" in df.columns:
    # contrib_df = df[df["Touches_File"].apply(as_bool)].copy()
    contrib_df = df.copy()
    print(f"Contribution rows touching tracked file: {len(contrib_df)} / {len(df)}")
else:
    contrib_df = df.copy()
    print("WARNING: 'Touches_File' column absent; all rows are treated as contributions.")

# Deduplicate once before expensive Git numstat extraction.
contrib_df = contrib_df.drop_duplicates(
    subset=["Key", "File_Path_std", commit_col]
).copy()

# Obtain per-commit/per-file change size directly from Git.
contrib_df = attach_change_size(contrib_df)

# Save diagnostic rows that could not be mapped to numstat.
unmatched_churn = contrib_df[contrib_df["Changed_Lines"].isna()].copy()
unmatched_churn.to_csv(OUT_CHURN_DIAGNOSTICS, index=False)
print(
    f"R4.8 churn mapping: valid={contrib_df['Changed_Lines'].notna().sum()} / {len(contrib_df)}; "
    f"unmatched={len(unmatched_churn)}"
)
print(f"Saved churn diagnostics: {OUT_CHURN_DIAGNOSTICS}")

# Numeric line-size columns
for c in ["Added_Lines", "Deleted_Lines", "Changed_Lines"]:
    contrib_df[c] = pd.to_numeric(contrib_df[c], errors="coerce")

# ================= MID-only =================
mids = contrib_df[contrib_df["Role_std"] == "MID"].copy()

mids = pd.merge(
    mids,
    authors[["Key", "File_Path_std", "self_author"]],
    on=["Key", "File_Path_std"],
    how="left"
)

mids["self_author"] = mids["self_author"].fillna("")
mids["is_self"]  = (mids["author_id"] == mids["self_author"])
mids["is_other"] = ~mids["is_self"]

mids["changed_by_self"] = np.where(mids["is_self"], mids["Changed_Lines"], 0.0)
mids["changed_by_other"] = np.where(mids["is_other"], mids["Changed_Lines"], 0.0)
mids["churn_missing"] = mids["Changed_Lines"].isna().astype(int)

per_file_mid = mids.groupby(["Key", "File_Path_std"]).agg(
    n_mid_total     = ("author_id", "size"),
    n_mid_by_self   = ("is_self", "sum"),
    n_mid_by_other  = ("is_other", "sum"),
    churn_mid_total = ("Changed_Lines", "sum"),
    churn_mid_by_self = ("changed_by_self", "sum"),
    churn_mid_by_other = ("changed_by_other", "sum"),
    churn_mid_missing_rows = ("churn_missing", "sum"),
).reset_index()

if not authors.empty:
    per_file_mid = (
        authors[["Key", "File_Path_std", "self_author"]]
        .merge(per_file_mid, on=["Key", "File_Path_std"], how="left")
    )
    for c in [
        "n_mid_total", "n_mid_by_self", "n_mid_by_other",
        "churn_mid_total", "churn_mid_by_self", "churn_mid_by_other",
        "churn_mid_missing_rows",
    ]:
        per_file_mid[c] = pd.to_numeric(per_file_mid[c], errors="coerce").fillna(0)

    per_file_mid[["n_mid_total", "n_mid_by_self", "n_mid_by_other", "churn_mid_missing_rows"]] = (
        per_file_mid[["n_mid_total", "n_mid_by_self", "n_mid_by_other", "churn_mid_missing_rows"]].astype(int)
    )

# ========== INTRO + MID + PAYMENT (full lifecycle range) ==========
range_df = contrib_df[contrib_df["Role_std"].isin(["INTRO", "MID", "PAYMENT"])].copy()

range_df = pd.merge(
    range_df,
    authors[["Key", "File_Path_std", "self_author"]],
    on=["Key", "File_Path_std"],
    how="left"
)

range_df["self_author"] = range_df["self_author"].fillna("")
range_df["is_self_author"]  = (range_df["author_id"] == range_df["self_author"])
range_df["is_other_author"] = ~range_df["is_self_author"]

range_df["changed_by_self"] = np.where(
    range_df["is_self_author"], range_df["Changed_Lines"], 0.0
)
range_df["changed_by_other"] = np.where(
    range_df["is_other_author"], range_df["Changed_Lines"], 0.0
)
range_df["churn_missing"] = range_df["Changed_Lines"].isna().astype(int)

per_file_range = range_df.groupby(["Key", "File_Path_std"]).agg(
    n_with_endpoints_total     = ("author_id", "size"),
    n_with_endpoints_by_self   = ("is_self_author", "sum"),
    n_with_endpoints_by_other  = ("is_other_author", "sum"),
    churn_with_endpoints_total = ("Changed_Lines", "sum"),
    churn_with_endpoints_by_self = ("changed_by_self", "sum"),
    churn_with_endpoints_by_other = ("changed_by_other", "sum"),
    churn_with_endpoints_missing_rows = ("churn_missing", "sum"),
).reset_index()

# ====== Merge MID + full range ======
per_file = pd.merge(
    per_file_mid,
    per_file_range,
    on=["Key", "File_Path_std"],
    how="outer"
)

# self_author may appear as self_author_x/self_author_y after merge
if "self_author_x" in per_file.columns:
    per_file["self_author"] = per_file["self_author_x"].fillna("")
    if "self_author_y" in per_file.columns:
        mask_empty = per_file["self_author"].eq("")
        per_file.loc[mask_empty, "self_author"] = per_file.loc[mask_empty, "self_author_y"].fillna("")
    per_file = per_file.drop(columns=[c for c in ["self_author_x", "self_author_y"] if c in per_file.columns])

for col in [
    "n_mid_total", "n_mid_by_self", "n_mid_by_other",
    "n_with_endpoints_total", "n_with_endpoints_by_self", "n_with_endpoints_by_other",
    "churn_mid_missing_rows", "churn_with_endpoints_missing_rows",
]:
    per_file[col] = pd.to_numeric(per_file[col], errors="coerce").fillna(0).astype(int)

for col in [
    "churn_mid_total", "churn_mid_by_self", "churn_mid_by_other",
    "churn_with_endpoints_total", "churn_with_endpoints_by_self", "churn_with_endpoints_by_other",
]:
    per_file[col] = pd.to_numeric(per_file[col], errors="coerce").fillna(0.0)

# ------------------------------------------------------------
# Commit-count shares (existing operationalization)
# ------------------------------------------------------------
per_file["n_mid_by_self/n_mid_total"] = np.where(
    per_file["n_mid_total"] > 0,
    per_file["n_mid_by_self"] / per_file["n_mid_total"],
    np.nan
)
per_file["n_mid_by_other/n_mid_total"] = np.where(
    per_file["n_mid_total"] > 0,
    per_file["n_mid_by_other"] / per_file["n_mid_total"],
    np.nan
)

per_file["n_with_endpoints_by_self/n_with_endpoints_total"] = np.where(
    per_file["n_with_endpoints_total"] > 0,
    per_file["n_with_endpoints_by_self"] / per_file["n_with_endpoints_total"],
    np.nan
)
per_file["n_with_endpoints_by_other/n_with_endpoints_total"] = np.where(
    per_file["n_with_endpoints_total"] > 0,
    per_file["n_with_endpoints_by_other"] / per_file["n_with_endpoints_total"],
    np.nan
)

# ------------------------------------------------------------
# R4.8 change-size shares (PRIMARY complementary measure)
# ------------------------------------------------------------
# We require complete numstat coverage within a file-item pair. If any
# contribution row is unmatched, the churn share is set to NaN rather than
# silently treating unknown churn as zero.
valid_mid_churn = (
    (per_file["churn_mid_total"] > 0) &
    (per_file["churn_mid_missing_rows"] == 0)
)
per_file["churn_mid_by_self/churn_mid_total"] = np.where(
    valid_mid_churn,
    per_file["churn_mid_by_self"] / per_file["churn_mid_total"],
    np.nan
)
per_file["churn_mid_by_other/churn_mid_total"] = np.where(
    valid_mid_churn,
    per_file["churn_mid_by_other"] / per_file["churn_mid_total"],
    np.nan
)

valid_full_churn = (
    (per_file["churn_with_endpoints_total"] > 0) &
    (per_file["churn_with_endpoints_missing_rows"] == 0)
)
per_file["churn_with_endpoints_by_self/churn_with_endpoints_total"] = np.where(
    valid_full_churn,
    per_file["churn_with_endpoints_by_self"] / per_file["churn_with_endpoints_total"],
    np.nan
)
per_file["churn_with_endpoints_by_other/churn_with_endpoints_total"] = np.where(
    valid_full_churn,
    per_file["churn_with_endpoints_by_other"] / per_file["churn_with_endpoints_total"],
    np.nan
)

# ----- Simpan CSV -----
per_file = per_file.rename(columns={"File_Path_std": "File_Path"})
cols_order = [
    "Key", "File_Path", "self_author",

    # commit counts
    "n_mid_total", "n_mid_by_self", "n_mid_by_other",
    "n_with_endpoints_total",
    "n_with_endpoints_by_self", "n_with_endpoints_by_other",

    # commit shares
    "n_mid_by_self/n_mid_total",
    "n_mid_by_other/n_mid_total",
    "n_with_endpoints_by_self/n_with_endpoints_total",
    "n_with_endpoints_by_other/n_with_endpoints_total",

    # R4.8: changed-line totals (added + deleted)
    "churn_mid_total", "churn_mid_by_self", "churn_mid_by_other",
    "churn_with_endpoints_total",
    "churn_with_endpoints_by_self", "churn_with_endpoints_by_other",

    # R4.8: changed-line shares
    "churn_mid_by_self/churn_mid_total",
    "churn_mid_by_other/churn_mid_total",
    "churn_with_endpoints_by_self/churn_with_endpoints_total",
    "churn_with_endpoints_by_other/churn_with_endpoints_total",

    # diagnostics
    "churn_mid_missing_rows",
    "churn_with_endpoints_missing_rows",
]
per_file = per_file[cols_order]

if FILTER_REQUIRE_MID:
    before = len(per_file)
    per_file = per_file[per_file["n_mid_total"] > 0].copy()
    print(f"Filter MID only: kept {len(per_file)} / {before} rows (removed {before - len(per_file)} without MID).")

per_file.to_csv(OUT_CSV, index=False)
print(f"Saved per-file summary WITH endpoints: {OUT_CSV}")

# ================== BOXPLOT COUNT (INTRO+MID+PAYMENT) ==================
if len(per_file) > 0:
    self_counts   = per_file["n_with_endpoints_by_self"].values
    others_counts = per_file["n_with_endpoints_by_other"].values

    fig, ax = plt.subplots(figsize=(7, 4))
    boxplot_multi_series(
        ax,
        [self_counts, others_counts],
        tick_labels=["Self-fixer", "Others"],
        y_label="Commits per file",
        title="",
        ylim=(0, None)
    )
    plt.tight_layout()
    plt.savefig(OUT_PDF_CNT, bbox_inches="tight")
    print(f"Saved boxplot COUNT (Self-fixer vs Others): {OUT_PDF_CNT}")
else:
    print("Tidak ada data per-file untuk boxplot COUNT.")

# ================== BOXPLOT RATIO (INTRO+MID+PAYMENT) ==================
rs = pd.to_numeric(
    per_file["n_with_endpoints_by_self/n_with_endpoints_total"],
    errors="coerce"
).dropna()

# IMPORTANT: others is deterministically 1 - self
ratio_self  = rs.to_numpy()
ratio_other = (1.0 - ratio_self)

if ratio_self.size:
    fig, ax = plt.subplots(figsize=(7, 4))
    boxplot_multi_series(
        ax,
        [ratio_self, ratio_other],
        tick_labels=["Self-fixer", "Others"],
        y_label="Share of commits per file",
        title="",
        ylim=(0, 1)
    )
    plt.tight_layout()
    plt.savefig(OUT_PDF_RATIO, bbox_inches="tight")
    print(f"Saved ratio boxplot (Self-fixer vs Others): {OUT_PDF_RATIO}")
else:
    print("Tidak ada data rasio untuk boxplot RATIO (INTRO+MID+PAYMENT).")

# ================== BOXPLOT RATIO (MID ONLY) ==================
rs_mid = pd.to_numeric(
    per_file["n_mid_by_self/n_mid_total"],
    errors="coerce"
).dropna()

ro_mid = 1.0 - rs_mid  # deterministic complement

# For the plot, you previously filtered >0; keep that if you want the same look,
# but do NOT use that filtered data for statistical tests.
ratio_self_mid_plot  = rs_mid[rs_mid > 0].to_numpy()
ratio_other_mid_plot = ro_mid[ro_mid > 0].to_numpy()

if ratio_self_mid_plot.size or ratio_other_mid_plot.size:
    fig, ax = plt.subplots(figsize=(8, 5))
    boxplot_multi_series(
        ax,
        [ratio_self_mid_plot, ratio_other_mid_plot],
        tick_labels=["Self-fixer", "Others"],
        y_label="Share of commits per file (MID only)",
        title="Share of commits during MID phase by role",
        ylim=(0, 1)
    )
    plt.tight_layout()
    plt.savefig(OUT_PDF_RATIO_MID, bbox_inches="tight")
    print(f"Saved MID ratio boxplot (Self-fixer vs Others): {OUT_PDF_RATIO_MID}")
else:
    print("Tidak ada data rasio untuk boxplot RATIO MID.")

# ==========================
# STATISTICAL TESTS (MOST APPROPRIATE)
# ==========================
# Test target:
#   H0: median(self_share) = 0.5
#   H1: median(self_share) > 0.5
#
# We run:
#   (A) Full range self_share (INTRO + MID + PAYMENT)
#   (B) MID-only self_share (recommended as less “by definition”)

full_x = pd.to_numeric(
    per_file["n_with_endpoints_by_self/n_with_endpoints_total"],
    errors="coerce"
).to_numpy()

mid_x = pd.to_numeric(
    per_file["n_mid_by_self/n_mid_total"],
    errors="coerce"
).to_numpy()

res_full = one_sample_wilcoxon_vs_mu0(full_x, mu0=0.5, alternative="greater")
res_mid  = one_sample_wilcoxon_vs_mu0(mid_x,  mu0=0.5, alternative="greater")

stats_df = pd.DataFrame([
    {"phase": "FULL_RANGE_INTRO_MID_PAYMENT", **res_full},
    {"phase": "MID_ONLY", **res_mid},
])

stats_df.to_csv(OUT_STATS_CSV, index=False)

lines = []
lines.append("=== One-sample tests against 0.5 (H1: self_share > 0.5) ===\n")
for _, r in stats_df.iterrows():
    lines.append(f"[{r['phase']}]")
    lines.append(f"n_total (non-NaN)  : {int(r['n_total'])}")
    lines.append(f"n_used (non-zero d): {int(r['n_used'])}")
    lines.append(f"median [q25,q75]   : {r['median']:.4f} [{r['q25']:.4f}, {r['q75']:.4f}]")
    lines.append(f"Pr(self_share > .5): {r['pr_gt_mu0']:.3f}")
    lines.append(f"Wilcoxon W         : {r['W']:.6g}")
    lines.append(f"Wilcoxon p         : {format_p(r['p'])}")
    lines.append(f"Effect size (RBC)  : {r['RBC']:.4f}")
    lines.append(f"Sign test p        : {format_p(r['sign_p'])}")
    lines.append("")

with open(OUT_STATS_TXT, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

print("\n".join(lines))
print(f"Saved stats CSV: {OUT_STATS_CSV}")
print(f"Saved stats TXT: {OUT_STATS_TXT}")


# ============================================================
# R4.8: BOXPLOTS FOR CHANGE-SIZE SHARE
# ============================================================
full_churn_self = pd.to_numeric(
    per_file["churn_with_endpoints_by_self/churn_with_endpoints_total"],
    errors="coerce"
).dropna()
full_churn_other = 1.0 - full_churn_self

if len(full_churn_self) > 0:
    fig, ax = plt.subplots(figsize=(7, 4))
    boxplot_multi_series(
        ax,
        [full_churn_self.to_numpy(), full_churn_other.to_numpy()],
        tick_labels=["Self-fixer", "Others"],
        y_label="Share of changed lines per file",
        title="",
        ylim=(0, 1)
    )
    plt.tight_layout()
    plt.savefig(OUT_PDF_CHURN_RATIO, bbox_inches="tight")
    print(f"Saved R4.8 change-size share boxplot: {OUT_PDF_CHURN_RATIO}")

mid_churn_self = pd.to_numeric(
    per_file["churn_mid_by_self/churn_mid_total"],
    errors="coerce"
).dropna()
mid_churn_other = 1.0 - mid_churn_self

if len(mid_churn_self) > 0:
    fig, ax = plt.subplots(figsize=(8, 5))
    boxplot_multi_series(
        ax,
        [mid_churn_self.to_numpy(), mid_churn_other.to_numpy()],
        tick_labels=["Self-fixer", "Others"],
        y_label="Share of changed lines per file (MID only)",
        title="Change-size share during MID phase by role",
        ylim=(0, 1)
    )
    plt.tight_layout()
    plt.savefig(OUT_PDF_CHURN_RATIO_MID, bbox_inches="tight")
    print(f"Saved R4.8 MID change-size share boxplot: {OUT_PDF_CHURN_RATIO_MID}")

# ============================================================
# R4.8: TEST COMMIT SHARE AND CHANGE-SIZE SHARE WITH SAME TEST
# ============================================================
commit_full_x = pd.to_numeric(
    per_file["n_with_endpoints_by_self/n_with_endpoints_total"],
    errors="coerce"
).to_numpy()
commit_mid_x = pd.to_numeric(
    per_file["n_mid_by_self/n_mid_total"],
    errors="coerce"
).to_numpy()
churn_full_x = pd.to_numeric(
    per_file["churn_with_endpoints_by_self/churn_with_endpoints_total"],
    errors="coerce"
).to_numpy()
churn_mid_x = pd.to_numeric(
    per_file["churn_mid_by_self/churn_mid_total"],
    errors="coerce"
).to_numpy()

r48_rows = []
for metric, phase, arr in [
    ("commit_share", "FULL_RANGE_INTRO_MID_PAYMENT", commit_full_x),
    ("commit_share", "MID_ONLY", commit_mid_x),
    ("change_size_share", "FULL_RANGE_INTRO_MID_PAYMENT", churn_full_x),
    ("change_size_share", "MID_ONLY", churn_mid_x),
]:
    rr = one_sample_wilcoxon_vs_mu0(arr, mu0=0.5, alternative="greater")
    r48_rows.append({"metric": metric, "phase": phase, **rr})

r48_stats = pd.DataFrame(r48_rows)
r48_stats.to_csv(OUT_STATS_R48_CSV, index=False)

# Robustness/concordance between frequency-based and magnitude-based shares
paired = per_file[[
    "n_with_endpoints_by_self/n_with_endpoints_total",
    "churn_with_endpoints_by_self/churn_with_endpoints_total"
]].apply(pd.to_numeric, errors="coerce").dropna()

if len(paired) >= 2:
    rho, rho_p = spearmanr(
        paired["n_with_endpoints_by_self/n_with_endpoints_total"],
        paired["churn_with_endpoints_by_self/churn_with_endpoints_total"]
    )
    commit_majority = paired["n_with_endpoints_by_self/n_with_endpoints_total"] > 0.5
    churn_majority = paired["churn_with_endpoints_by_self/churn_with_endpoints_total"] > 0.5
    majority_agreement = float(np.mean(commit_majority == churn_majority))
    median_abs_diff = float(np.median(np.abs(
        paired["n_with_endpoints_by_self/n_with_endpoints_total"] -
        paired["churn_with_endpoints_by_self/churn_with_endpoints_total"]
    )))
else:
    rho = rho_p = majority_agreement = median_abs_diff = np.nan

r48_lines = []
r48_lines.append("=== R4.8: Commit-share vs change-size-share analysis ===\n")
r48_lines.append("Change size = added lines + deleted lines (per commit, per tracked file).")
r48_lines.append("Primary R4.8 measure: each role's share of total changed lines on the file.\n")

for _, r in r48_stats.iterrows():
    r48_lines.append(f"[{r['metric']} | {r['phase']}]")
    r48_lines.append(f"n_total (non-NaN)   : {int(r['n_total'])}")
    r48_lines.append(f"n_used (non-zero d) : {int(r['n_used'])}")
    r48_lines.append(f"median [q25,q75]    : {r['median']:.4f} [{r['q25']:.4f}, {r['q75']:.4f}]")
    r48_lines.append(f"Pr(self_share > .5) : {r['pr_gt_mu0']:.3f}")
    r48_lines.append(f"Wilcoxon W          : {r['W']:.6g}")
    r48_lines.append(f"Wilcoxon p          : {format_p(r['p'])}")
    r48_lines.append(f"Effect size (RBC)   : {r['RBC']:.4f}")
    r48_lines.append(f"Sign test p         : {format_p(r['sign_p'])}")
    r48_lines.append("")

r48_lines.append("[Robustness: full-range commit share vs change-size share]")
r48_lines.append(f"paired file-item pairs : {len(paired)}")
r48_lines.append(f"Spearman rho           : {rho:.4f}")
r48_lines.append(f"Spearman p             : {format_p(rho_p)}")
r48_lines.append(f"majority agreement     : {majority_agreement:.3f}")
r48_lines.append(f"median absolute diff   : {median_abs_diff:.4f}")
r48_lines.append("")
r48_lines.append(
    f"numstat coverage       : {contrib_df['Changed_Lines'].notna().sum()} / {len(contrib_df)} contribution rows"
)
r48_lines.append(f"unmatched rows          : {len(unmatched_churn)}")

with open(OUT_STATS_R48_TXT, "w", encoding="utf-8") as f:
    f.write("\n".join(r48_lines))

print("\n".join(r48_lines))
print(f"Saved R4.8 stats CSV: {OUT_STATS_R48_CSV}")
print(f"Saved R4.8 stats TXT: {OUT_STATS_R48_TXT}")
