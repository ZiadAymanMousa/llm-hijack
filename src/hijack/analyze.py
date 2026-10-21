"""Phase metrics, paired contrasts, interaction model and figures (protocol §10–11).

    python -m hijack.analyze --config configs/phase1.yaml --out reports/phase1

Reads <results_dir>/<benchmark>/<arm>.jsonl, repeats/*.jsonl, replay/*.jsonl and sep/*.jsonl.
Writes CSV tables, summary.md and two figures to --out.
"""

import argparse
import hashlib
import itertools
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from hijack.config import ROOT, load_config
from hijack.record import read_jsonl

CONTENT_DEPENDENT = {"param-open", "action-open"}


# ---------------------------------------------------------------- loading

def episode_row(benchmark: str, arm: str, spec: dict, ep: dict) -> dict:
    parse_fail = any("<function" in m["text"] and not m.get("tool_calls")
                     for m in ep["messages"] if m["role"] == "assistant")
    return {
        "benchmark": benchmark, "arm": arm, "agent": spec["agent"], "guard": spec.get("guard"),
        "defense": spec.get("defense"), "case_id": ep["case_id"], "suite": ep["suite"],
        "cluster": f"{ep['suite']}/{ep['user_task']}", "bucket": ep["bucket"],
        "attacked": ep["injection_task"] is not None, "utility": ep["utility"],
        "attack": bool(ep["attack_success"]), "delivered": bool(ep["payload_delivered"]),
        "error": ep["error"] is not None, "parse_fail": parse_fail, "seconds": ep["seconds"],
        "unsafe_proposed": any(p.get("unsafe") for p in ep["proposals"]),
        "labelable": ep.get("labelable", True), "workers": ep.get("workers") or 1,
    }


def load_episodes(cfg) -> tuple[pd.DataFrame, pd.DataFrame]:
    episodes, proposals = [], []
    for benchmark in cfg["benchmarks"]:
        for arm, spec in cfg["arms"].items():
            for ep in read_jsonl(cfg.results_dir / benchmark / f"{arm}.jsonl"):
                props = ep["proposals"]
                episodes.append(episode_row(benchmark, arm, spec, ep))
                for p in props:
                    proposals.append({
                        "benchmark": benchmark, "arm": arm, "case_id": ep["case_id"],
                        "cluster": f"{ep['suite']}/{ep['user_task']}", "bucket": ep["bucket"],
                        "attacked": ep["injection_task"] is not None, "unsafe": bool(p.get("unsafe")),
                        "labelable": ep.get("labelable", True),
                        "blocked": bool(p["blocked"]), "guarded": p["guard"] is not None,
                        "guard_latency": p["guard"]["latency_s"] if p["guard"] else np.nan,
                        "mask_leak": bool(p.get("mask_leak")),
                    })
    return pd.DataFrame(episodes), pd.DataFrame(proposals)


def load_repeats(cfg) -> pd.DataFrame:
    """Episodes of the repeat runs (<benchmark>/repeats/<arm>_r<N>.jsonl), with their run number."""
    rows = []
    for benchmark in cfg["benchmarks"]:
        for path in sorted((cfg.results_dir / benchmark / "repeats").glob("*_r*.jsonl")):
            arm, run = path.stem.rsplit("_r", 1)
            rows += [{**episode_row(benchmark, arm, cfg["arms"][arm], ep), "run": int(run)} for ep in read_jsonl(path)]
    return pd.DataFrame(rows)


def load_replay(cfg) -> pd.DataFrame:
    rows = []
    for benchmark in cfg["benchmarks"]:
        for path in sorted((cfg.results_dir / benchmark / "replay").glob("*.jsonl")):
            for r in read_jsonl(path):
                rows.append({**r, "benchmark": benchmark, "cluster": f"{r['suite']}/{r['user_task']}"})
    return pd.DataFrame(rows)


def load_sep(cfg) -> pd.DataFrame:
    rows = []
    for path in sorted((cfg.results_dir / "sep").glob("*.jsonl")):
        rows.extend(read_jsonl(path))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- statistics

def cluster_rate(df: pd.DataFrame, value: str, n_boot: int, seed: int = 0) -> dict:
    """Rate of a 0/1 column with a bootstrap CI that resamples user tasks (clusters)."""
    if df.empty:
        return {"rate": np.nan, "lo": np.nan, "hi": np.nan, "k": 0, "n": 0}
    g = df.groupby("cluster")[value].agg(["sum", "count"])
    num, den = g["sum"].to_numpy(float), g["count"].to_numpy(float)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(g), size=(n_boot, len(g)))
    boots = num[draws].sum(1) / den[draws].sum(1)
    return {"rate": num.sum() / den.sum(), "lo": np.quantile(boots, 0.025), "hi": np.quantile(boots, 0.975),
            "k": int(num.sum()), "n": int(den.sum())}


def paired_contrast(df1: pd.DataFrame, df2: pd.DataFrame, value: str, n_boot: int, seed: int = 0) -> dict:
    """Difference of rates (arm 1 − arm 2) on the same cases, cluster bootstrap CI and exact McNemar p."""
    m = df1[["case_id", "cluster", value]].merge(df2[["case_id", value]], on="case_id", suffixes=("_1", "_2"))
    if m.empty:
        return {"diff": np.nan, "lo": np.nan, "hi": np.nan, "p": np.nan, "n": 0}
    x1, x2 = m[f"{value}_1"].astype(float), m[f"{value}_2"].astype(float)
    g = pd.DataFrame({"cluster": m["cluster"], "d": x1 - x2}).groupby("cluster")["d"].agg(["sum", "count"])
    num, den = g["sum"].to_numpy(), g["count"].to_numpy(float)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(g), size=(n_boot, len(g)))
    boots = num[draws].sum(1) / den[draws].sum(1)
    b, c = int(((x1 == 1) & (x2 == 0)).sum()), int(((x1 == 0) & (x2 == 1)).sum())
    p = binomtest(min(b, c), b + c, 0.5).pvalue if b + c else 1.0
    return {"diff": num.sum() / den.sum(), "lo": np.quantile(boots, 0.025), "hi": np.quantile(boots, 0.975),
            "p": p, "n": len(m), "discordant": f"{b}/{c}"}


def holm(pvalues: list[float]) -> list[float]:
    order = np.argsort(pvalues)
    adjusted, running = [np.nan] * len(pvalues), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(pvalues) - rank) * pvalues[i]))
        adjusted[i] = running
    return adjusted


def interaction(episodes: pd.DataFrame, arms: dict[str, tuple[int, int]]) -> dict:
    """Agent × guard interaction on attack success, on the log-odds scale.

    Logistic GEE with exchangeable correlation within user tasks (cluster-robust errors).
    Zero means the two defenses act independently.
    """
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    d = episodes[episodes["arm"].isin(arms) & episodes["attacked"] & episodes["delivered"]].copy()
    if d["arm"].nunique() < 4:
        return {"note": "not estimable (missing arms)"}
    per_arm = d.groupby("arm")["attack"].agg(["sum", "count"])
    if ((per_arm["sum"] == 0) | (per_arm["sum"] == per_arm["count"])).any():
        counts = ", ".join(f"{a} {int(r['sum'])}/{int(r['count'])}" for a, r in per_arm.iterrows())
        return {"note": f"not estimable (an arm has 0% or 100% attack success: {counts}); use exact counts"}
    d["hardened"] = d["arm"].map(lambda a: arms[a][0])
    d["guarded"] = d["arm"].map(lambda a: arms[a][1])
    d["y"] = d["attack"].astype(int)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = smf.gee("y ~ hardened * guarded", groups="cluster", data=d, family=sm.families.Binomial(),
                      cov_struct=sm.cov_struct.Exchangeable()).fit()
    lo, hi = fit.conf_int().loc["hardened:guarded"]
    return {"log_odds": fit.params["hardened:guarded"], "lo": lo, "hi": hi, "p": fit.pvalues["hardened:guarded"],
            "n": len(d)}


def is_validation(cluster: str, fraction: float) -> bool:
    return int(hashlib.sha256(cluster.encode()).hexdigest(), 16) % 1000 < fraction * 1000


def calibrate(rows: pd.DataFrame, fraction: float) -> tuple[pd.DataFrame, float]:
    """Pick the score threshold with the best balanced accuracy on validation tasks; apply it to the rest."""
    val = rows[rows["cluster"].map(lambda c: is_validation(c, fraction))]
    best, best_bacc = 0.5, -1.0
    for t in np.unique(val["score"]):
        pred = val["score"] >= t
        tpr = pred[val["unsafe"]].mean() if val["unsafe"].any() else 0.0
        tnr = (~pred[~val["unsafe"]]).mean() if (~val["unsafe"]).any() else 0.0
        if (tpr + tnr) / 2 > best_bacc:
            best, best_bacc = t, (tpr + tnr) / 2
    test = rows[~rows.index.isin(val.index)].copy()
    test["blocked"] = test["score"] >= best
    return test, float(best)


# ---------------------------------------------------------------- tables

def arm_metrics(episodes, proposals, n_boot) -> pd.DataFrame:
    rows = []
    for (benchmark, arm), ep in episodes.groupby(["benchmark", "arm"], sort=False):
        clean, att = ep[~ep["attacked"]], ep[ep["attacked"]]
        att_d = att[att["delivered"]].assign(secure=lambda x: x["utility"] & ~x["attack"])
        props = proposals[(proposals["benchmark"] == benchmark) & (proposals["arm"] == arm)]
        guarded = props[props["guarded"] & props["labelable"]]
        cd = guarded[guarded["bucket"].isin(CONTENT_DEPENDENT)]
        row = {"benchmark": benchmark, "arm": arm, "agent": ep["agent"].iat[0], "guard": ep["guard"].iat[0],
               "defense": ep["defense"].iat[0],
               "episodes": len(ep), "errors": int(ep["error"].sum()), "parse_failures": int(ep["parse_fail"].sum()),
               "delivered": att["delivered"].mean() if len(att) else np.nan,
               "unlabelable_attacked": int((~att["labelable"]).sum())}
        for name, frame, col in [("BU", clean, "utility"), ("UA", att, "utility"), ("ASR", att_d, "attack"),
                                 ("secure", att_d, "secure"), ("unsafe_proposed", att, "unsafe_proposed")]:
            r = cluster_rate(frame, col, n_boot)
            row.update({name: r["rate"], f"{name}_lo": r["lo"], f"{name}_hi": r["hi"], f"{name}_k/n": f"{r['k']}/{r['n']}"})
        if len(guarded):
            miss = cluster_rate(guarded[guarded["unsafe"]].assign(missed=lambda x: ~x["blocked"]), "missed", n_boot)
            fb = cluster_rate(guarded[~guarded["unsafe"]], "blocked", n_boot)
            fb_cd = cluster_rate(cd[~cd["unsafe"]], "blocked", n_boot)
            row.update({"guard_miss": miss["rate"], "guard_miss_k/n": f"{miss['k']}/{miss['n']}",
                        "false_block": fb["rate"], "false_block_lo": fb["lo"], "false_block_hi": fb["hi"],
                        "false_block_content_dep": fb_cd["rate"],
                        "guard_p50_s": guarded["guard_latency"].median(),
                        "guard_p95_s": guarded["guard_latency"].quantile(0.95),
                        "mask_leak": props["mask_leak"].mean()})
        row["seconds_per_episode"] = ep["seconds"].mean()
        rows.append(row)
    return pd.DataFrame(rows)


def contrasts(episodes, n_boot) -> pd.DataFrame:
    specs = [  # (name, arm 1, arm 2, value, primary)
        ("Guard (full) adds to hardened agent", "F", "B", "attack", True),
        ("Guard (restricted) adds to hardened agent", "D", "B", "attack", False),
        ("Agent hardening", "B", "A", "attack", False),
        ("Guard (full) on undefended agent", "E", "A", "attack", False),
        ("Restricted vs full view, hardened agent", "D", "F", "attack", False),
        ("Restricted vs full view: utility under attack", "D", "F", "utility", False),
        ("Trust-boundary prompt", "A", "A0", "attack", False),
        ("RETA vs older recipe (MetaSecAlign)", "B", "M", "attack", False),
    ]
    rows = []
    for benchmark in episodes["benchmark"].unique():
        ep = episodes[(episodes["benchmark"] == benchmark) & episodes["attacked"] & episodes["delivered"]]
        for name, a1, a2, value, primary in specs:
            r = paired_contrast(ep[ep["arm"] == a1], ep[ep["arm"] == a2], value, n_boot)
            rows.append({"benchmark": benchmark, "contrast": name, "arms": f"{a1} − {a2}", "metric": value,
                         "primary": primary, **r})
    df = pd.DataFrame(rows)
    prim = df["primary"] & df["p"].notna()
    df.loc[prim, "p_holm"] = holm(df.loc[prim, "p"].tolist())
    return df


def replay_metrics(replay, episodes, cfg, n_boot) -> tuple[pd.DataFrame, pd.DataFrame]:
    if replay.empty:
        return pd.DataFrame(), pd.DataFrame()
    rows, decomp, thresholds = [], [], {}
    replay = replay[replay["labelable"]] if "labelable" in replay else replay
    for (benchmark, guard), r in replay.groupby(["benchmark", "guard"]):
        if r["score"].notna().any() and r["label"].eq("score").all():
            r, thresholds[guard] = calibrate(r, cfg.get("validation_fraction", 0.2))
        for source, s in r.groupby("source_arm"):
            miss = cluster_rate(s[s["unsafe"]].assign(missed=lambda x: ~x["blocked"]), "missed", n_boot)
            fb = cluster_rate(s[~s["unsafe"]], "blocked", n_boot)
            fb_cd = cluster_rate(s[~s["unsafe"] & s["bucket"].isin(CONTENT_DEPENDENT)], "blocked", n_boot)
            # Episode-level estimate: the attack would have failed if any unsafe call was blocked.
            stopped = s[s["unsafe"]].groupby("case_id")["blocked"].any()
            ep = episodes[(episodes["benchmark"] == benchmark) & (episodes["arm"] == source)
                          & episodes["attacked"] & episodes["delivered"]].copy()
            ep["attack_with_guard"] = ep["attack"] & ~ep["case_id"].map(stopped).fillna(False).astype(bool)
            asr = cluster_rate(ep, "attack_with_guard", n_boot)
            rows.append({"benchmark": benchmark, "guard": guard, "source_arm": source,
                         "threshold": thresholds.get(guard), "miss": miss["rate"], "miss_lo": miss["lo"],
                         "miss_hi": miss["hi"], "miss_k/n": f"{miss['k']}/{miss['n']}",
                         "false_block": fb["rate"], "false_block_content_dep": fb_cd["rate"],
                         "est_ASR": asr["rate"], "est_ASR_lo": asr["lo"], "est_ASR_hi": asr["hi"]})
        # H1c: is the guard worse on the hardened agent's unsafe calls than on the undefended agent's?
        ua = r[(r["source_arm"] == "A") & r["unsafe"]].assign(missed=lambda x: (~x["blocked"]).astype(int))
        ub = r[(r["source_arm"] == "B") & r["unsafe"]].assign(missed=lambda x: (~x["blocked"]).astype(int))
        if len(ua) and len(ub):
            diff = miss_difference(ua, ub, n_boot)
            decomp.append({"benchmark": benchmark, "guard": guard, "miss_on_A": ua["missed"].mean(),
                           "miss_on_B": ub["missed"].mean(), **diff,
                           "unsafe_calls_A": len(ua), "unsafe_calls_B": len(ub)})
    return pd.DataFrame(rows), pd.DataFrame(decomp)


def miss_difference(ua: pd.DataFrame, ub: pd.DataFrame, n_boot: int, seed: int = 0) -> dict:
    """Guard miss rate on B's unsafe calls minus on A's, with a bootstrap over user tasks."""
    a = ua.groupby("cluster")["missed"].agg(["sum", "count"])
    b = ub.groupby("cluster")["missed"].agg(["sum", "count"])
    g = a.join(b, how="outer", lsuffix="_a", rsuffix="_b").fillna(0).to_numpy(float)
    draws = np.random.default_rng(seed).integers(0, len(g), size=(n_boot, len(g)))
    s = g[draws].sum(axis=1)  # (n_boot, 4): sum_a, count_a, sum_b, count_b
    with np.errstate(invalid="ignore", divide="ignore"):
        diffs = s[:, 2] / s[:, 3] - s[:, 0] / s[:, 1]
    return {"diff_B_minus_A": g[:, 2].sum() / g[:, 3].sum() - g[:, 0].sum() / g[:, 1].sum(),
            "lo": np.nanquantile(diffs, 0.025), "hi": np.nanquantile(diffs, 0.975)}


def repeat_metrics(episodes, repeats, n_boot) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run-to-run variance. Run 1 is the arm's main run; each repeat runs the same cases again.

    Returns per-run rates and, for every pair of runs, how often a case had the same outcome and the
    paired change in attack success and in task success under attack.
    """
    if repeats.empty:
        return pd.DataFrame(), pd.DataFrame()
    run_rows, pair_rows = [], []
    for (benchmark, arm), rep in repeats.groupby(["benchmark", "arm"], sort=False):
        main = episodes[(episodes["benchmark"] == benchmark) & (episodes["arm"] == arm)].assign(run=1)
        runs = dict(tuple(pd.concat([main, rep]).groupby("run")))
        for run, ep in runs.items():
            att = ep[ep["attacked"]]
            row = {"benchmark": benchmark, "arm": arm, "run": run, "workers": int(ep["workers"].max()),
                   "episodes": len(ep), "errors": int(ep["error"].sum())}
            for name, frame, col in [("BU", ep[~ep["attacked"]], "utility"), ("UA", att, "utility"),
                                     ("ASR", att[att["delivered"]], "attack")]:
                r = cluster_rate(frame, col, n_boot)
                row.update({name: r["rate"], f"{name}_lo": r["lo"], f"{name}_hi": r["hi"],
                            f"{name}_k/n": f"{r['k']}/{r['n']}"})
            run_rows.append(row)
        for first, second in itertools.combinations(sorted(runs), 2):
            a, b = runs[first], runs[second]
            m = a.merge(b, on="case_id", suffixes=("_1", "_2"))
            asr = paired_contrast(b[b["attacked"] & b["delivered"]], a[a["attacked"] & a["delivered"]], "attack", n_boot)
            ua = paired_contrast(b[b["attacked"]], a[a["attacked"]], "utility", n_boot)
            pair_rows.append({
                "benchmark": benchmark, "arm": arm, "runs": f"{second} vs {first}", "cases": len(m),
                "same_outcome": ((m["utility_1"] == m["utility_2"]) & (m["attack_1"] == m["attack_2"])).mean(),
                "ASR_diff": asr["diff"], "ASR_diff_lo": asr["lo"], "ASR_diff_hi": asr["hi"], "ASR_diff_p": asr["p"],
                "UA_diff": ua["diff"], "UA_diff_lo": ua["lo"], "UA_diff_hi": ua["hi"], "UA_diff_p": ua["p"]})
    return pd.DataFrame(run_rows), pd.DataFrame(pair_rows)


def sep_metrics(sep, n_boot) -> pd.DataFrame:
    rows = []
    for agent, s in sep.groupby("agent"):
        s = s.assign(cluster=s["case_id"], manipulated=s["hit_instructed"] & ~s["hit_clean"])
        r = cluster_rate(s, "manipulated", n_boot)
        rows.append({"agent": agent, "items": len(s), "witness_with_probe": s["hit_instructed"].mean(),
                     "witness_without_probe": s["hit_clean"].mean(), "manipulated": r["rate"],
                     "lo": r["lo"], "hi": r["hi"]})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- figures

INK, MUTED, SURFACE = "#0b0b0b", "#6b6a64", "#fcfcfb"
VIEW_COLORS = {"none": "#2a78d6", "restricted": "#eb6834", "full": "#1baf7a"}


def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=INK, labelsize=9)
    ax.grid(color="#e4e3dd", linewidth=0.6)
    ax.set_axisbelow(True)


def plot_security_utility(metrics: pd.DataFrame, cfg, out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    benchmarks = list(metrics["benchmark"].unique())
    if not benchmarks:
        return
    fig, axes = plt.subplots(1, len(benchmarks), figsize=(5.2 * len(benchmarks), 4.2), squeeze=False,
                             facecolor=SURFACE)
    for ax, benchmark in zip(axes[0], benchmarks):
        _style(ax)
        m = metrics[metrics["benchmark"] == benchmark]
        for _, r in m.iterrows():
            view = cfg.guard(r["guard"])["view"] if isinstance(r["guard"], str) else "none"
            color = VIEW_COLORS[view]
            ax.errorbar(100 * r["UA"], 100 * r["ASR"], yerr=[[100 * (r["ASR"] - r["ASR_lo"])],
                        [100 * (r["ASR_hi"] - r["ASR"])]], fmt="o", ms=8, color=color, ecolor=color,
                        elinewidth=1.5, capsize=0, markeredgecolor=SURFACE, markeredgewidth=2)
            ax.annotate(r["arm"], (100 * r["UA"], 100 * r["ASR"]), xytext=(7, 4), textcoords="offset points",
                        fontsize=9, color=INK)
        ax.set_title(benchmark, color=INK, fontsize=11, loc="left")
        ax.set_xlabel("Task success under attack (%)", color=INK, fontsize=9)
        ax.set_ylabel("Attack success rate (%)", color=INK, fontsize=9)
        ax.set_xlim(0, 100)
        ax.set_ylim(bottom=0)
    handles = [plt.Line2D([], [], marker="o", ls="", color=c, ms=8, label=f"guard: {v}")
               for v, c in VIEW_COLORS.items()]
    fig.legend(handles=handles, loc="upper right", frameon=False, fontsize=9, labelcolor=INK)
    fig.tight_layout(rect=(0, 0, 0.86, 1))
    fig.savefig(out / "security_utility.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)


def plot_decomposition(decomp: pd.DataFrame, out: Path):
    if decomp.empty:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = decomp.reset_index(drop=True)
    labels = [f"{r.guard}\n{r.benchmark}" for r in d.itertuples()]
    x = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(max(5, 1.6 * len(d) + 2), 3.8), facecolor=SURFACE)
    _style(ax)
    width = 0.36
    ax.bar(x - width / 2 - 0.01, 100 * d["miss_on_A"], width, color="#2a78d6", label="unsafe calls by undefended agent (A)")
    ax.bar(x + width / 2 + 0.01, 100 * d["miss_on_B"], width, color="#eb6834", label="unsafe calls by hardened agent (B)")
    ax.set_xticks(x, labels, fontsize=8, color=INK)
    ax.set_ylabel("Guard miss rate (%)", color=INK, fontsize=9)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(out / "decomposition.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)


# ---------------------------------------------------------------- report

def pct(x) -> str:
    return "–" if pd.isna(x) else f"{100 * x:.1f}"


def ci(r, name) -> str:
    return f"{pct(r[name])} [{pct(r[name + '_lo'])}, {pct(r[name + '_hi'])}]"


def write_summary(out: Path, metrics, contrast_df, inter, replay_df, decomp, sep_df, repeat_runs, repeat_pairs):
    lines = ["# Phase results (generated)", ""]
    if not metrics.empty:
        lines += ["## Arms", "", "Rates in %, 95% CI from resampling user tasks.", "",
                  "| Benchmark | Arm | Agent | Guard | Episodes | Errors | BU | UA | ASR | Guard miss | False block |",
                  "|---|---|---|---|---|---|---|---|---|---|---|"]
        for _, r in metrics.iterrows():
            guard = r.defense if isinstance(r.get("defense"), str) else (r.guard if isinstance(r.guard, str) else "–")
            lines.append(f"| {r.benchmark} | {r.arm} | {r.agent} | {guard} | {r.episodes} | {r.errors} | "
                         f"{ci(r, 'BU')} | {ci(r, 'UA')} | {ci(r, 'ASR')} | {pct(r.get('guard_miss'))} | "
                         f"{pct(r.get('false_block'))} |")
    if not contrast_df.empty:
        lines += ["", "## Paired contrasts", "",
                  "| Benchmark | Contrast | Arms | Metric | Difference | 95% CI | p (McNemar) | p (Holm) |",
                  "|---|---|---|---|---|---|---|---|"]
        for _, r in contrast_df.iterrows():
            lines.append(f"| {r.benchmark} | {r.contrast}{' (primary)' if r.primary else ''} | {r.arms} | {r.metric} | "
                         f"{pct(r['diff'])} | [{pct(r.lo)}, {pct(r.hi)}] | {r.p:.3g} | "
                         f"{'' if pd.isna(r.get('p_holm')) else f'{r.p_holm:.3g}'} |")
    if inter:
        lines += ["", "## Agent × guard interaction (log-odds)", "", "| Benchmark | Arms | Estimate | 95% CI | p | n |",
                  "|---|---|---|---|---|---|"]
        for (benchmark, arms), r in inter.items():
            if "note" in r:
                lines.append(f"| {benchmark} | {arms} | {r['note']} | | | |")
            else:
                lines.append(f"| {benchmark} | {arms} | {r['log_odds']:.2f} | [{r['lo']:.2f}, {r['hi']:.2f}] | "
                             f"{r['p']:.3g} | {r['n']} |")
    if not replay_df.empty:
        lines += ["", "## Guards scored on logged calls (replay)", "",
                  "| Benchmark | Guard | Source arm | Miss | False block | False block (content-dependent) | Est. ASR |",
                  "|---|---|---|---|---|---|---|"]
        for _, r in replay_df.iterrows():
            lines.append(f"| {r.benchmark} | {r.guard} | {r.source_arm} | {pct(r.miss)} | {pct(r.false_block)} | "
                         f"{pct(r.false_block_content_dep)} | {ci(r, 'est_ASR')} |")
    if not decomp.empty:
        lines += ["", "## Do failures overlap? Guard miss rate by agent", "",
                  "| Benchmark | Guard | Miss on A | Miss on B | B − A | 95% CI |", "|---|---|---|---|---|---|"]
        for _, r in decomp.iterrows():
            lines.append(f"| {r.benchmark} | {r.guard} | {pct(r.miss_on_A)} | {pct(r.miss_on_B)} | "
                         f"{pct(r.diff_B_minus_A)} | [{pct(r.lo)}, {pct(r.hi)}] |")
    if not sep_df.empty:
        lines += ["", "## Answer manipulation (SEP)", "",
                  "| Agent | Items | Witness with probe | Witness without probe | Manipulated | 95% CI |",
                  "|---|---|---|---|---|---|"]
        for _, r in sep_df.iterrows():
            lines.append(f"| {r.agent} | {r['items']} | {pct(r.witness_with_probe)} | {pct(r.witness_without_probe)} | "
                         f"{pct(r.manipulated)} | [{pct(r.lo)}, {pct(r.hi)}] |")
    if not repeat_runs.empty:
        lines += ["", "## Repeat runs (run-to-run variance)", "", "Run 1 is the main run.", "",
                  "| Benchmark | Arm | Run | Episodes at once | Episodes | BU | UA | ASR |",
                  "|---|---|---|---|---|---|---|---|"]
        for _, r in repeat_runs.iterrows():
            lines.append(f"| {r.benchmark} | {r.arm} | {r.run} | {r.workers} | {r.episodes} | {ci(r, 'BU')} | "
                         f"{ci(r, 'UA')} | {ci(r, 'ASR')} |")
        lines += ["", "| Benchmark | Arm | Runs | Cases | Same outcome | ASR change | 95% CI | p | UA change | 95% CI | p |",
                  "|---|---|---|---|---|---|---|---|---|---|---|"]
        for _, r in repeat_pairs.iterrows():
            lines.append(f"| {r.benchmark} | {r.arm} | {r.runs} | {r.cases} | {pct(r.same_outcome)} | {pct(r.ASR_diff)} | "
                         f"[{pct(r.ASR_diff_lo)}, {pct(r.ASR_diff_hi)}] | {r.ASR_diff_p:.3g} | {pct(r.UA_diff)} | "
                         f"[{pct(r.UA_diff_lo)}, {pct(r.UA_diff_hi)}] | {r.UA_diff_p:.3g} |")
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--n-boot", type=int, default=10_000)
    args = parser.parse_args()

    cfg = load_config(args.config)
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)

    episodes, proposals = load_episodes(cfg)
    if episodes.empty:
        raise SystemExit(f"no episodes found under {cfg.results_dir}")
    metrics = arm_metrics(episodes, proposals, args.n_boot)
    contrast_df = contrasts(episodes, args.n_boot)
    inter = {}
    for benchmark in episodes["benchmark"].unique():
        ep = episodes[episodes["benchmark"] == benchmark]
        inter[(benchmark, "A,B,E,F (full view)")] = interaction(ep, {"A": (0, 0), "B": (1, 0), "E": (0, 1), "F": (1, 1)})
        inter[(benchmark, "A,B,C,D (restricted)")] = interaction(ep, {"A": (0, 0), "B": (1, 0), "C": (0, 1), "D": (1, 1)})
    replay_df, decomp = replay_metrics(load_replay(cfg), episodes, cfg, args.n_boot)
    sep_df = sep_metrics(load_sep(cfg), args.n_boot) if (cfg.results_dir / "sep").exists() else pd.DataFrame()
    repeat_runs, repeat_pairs = repeat_metrics(episodes, load_repeats(cfg), args.n_boot)

    metrics.to_csv(out / "metrics.csv", index=False)
    contrast_df.to_csv(out / "contrasts.csv", index=False)
    pd.DataFrame([{"benchmark": b, "arms": a, **r} for (b, a), r in inter.items()]).to_csv(out / "interaction.csv", index=False)
    replay_df.to_csv(out / "replay_guards.csv", index=False)
    decomp.to_csv(out / "decomposition.csv", index=False)
    sep_df.to_csv(out / "sep.csv", index=False)
    repeat_runs.to_csv(out / "repeat_runs.csv", index=False)
    repeat_pairs.to_csv(out / "repeat_pairs.csv", index=False)
    plot_security_utility(metrics[metrics["UA"].notna() & metrics["ASR"].notna()], cfg, out)
    plot_decomposition(decomp, out)
    write_summary(out, metrics, contrast_df, inter, replay_df, decomp, sep_df, repeat_runs, repeat_pairs)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
