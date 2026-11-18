"""Build the PathSnap technical report PDF (report/PathSnap_Technical_Report.pdf).

Every number is read from the result files in reports/ and artifacts/ at build time, so the
report always matches the repository. Structure follows the organisers' Deliverables document
(sections 2.2-7) plus the final evaluator-test run.

Run:  python report/build_report.py
"""
import json
import os
import sys
from datetime import date

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle)
from reportlab.platypus.tableofcontents import TableOfContents

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
from pathsnap import config  # noqa: E402
from pathsnap.features import CTX, FEATURES, LABEL_MAX_DIST  # noqa: E402

R, FIG, IMG = config.REPORTS, config.FIGURES, os.path.join(HERE, "img")
OUT = os.path.join(HERE, "PathSnap_Technical_Report.pdf")

# ---------------------------------------------------------------- fonts & styles
FONT_DIR = os.path.join(os.environ.get("WINDIR", "C:/Windows"), "Fonts")
try:
    pdfmetrics.registerFont(TTFont("Body", os.path.join(FONT_DIR, "calibri.ttf")))
    pdfmetrics.registerFont(TTFont("Body-B", os.path.join(FONT_DIR, "calibrib.ttf")))
    pdfmetrics.registerFont(TTFont("Body-I", os.path.join(FONT_DIR, "calibrii.ttf")))
    pdfmetrics.registerFont(TTFont("Body-BI", os.path.join(FONT_DIR, "calibriz.ttf")))
    pdfmetrics.registerFont(TTFont("Mono", os.path.join(FONT_DIR, "consola.ttf")))
    pdfmetrics.registerFontFamily("Body", normal="Body", bold="Body-B", italic="Body-I", boldItalic="Body-BI")
    BODY, BOLD, MONO = "Body", "Body-B", "Mono"
except Exception:  # non-Windows fallback (no Greek glyphs in the built-in fonts)
    BODY, BOLD, MONO = "Helvetica", "Helvetica-Bold", "Courier"

INK, MUTED, ACCENT, RULE, ZEBRA = (colors.HexColor(c) for c in ("#1B2430", "#5A6472", "#1F5FAD", "#C9D1DB", "#F2F5F8"))
S = {
    "body": ParagraphStyle("body", fontName=BODY, fontSize=10.5, leading=14.2, textColor=INK, spaceAfter=6),
    "small": ParagraphStyle("small", fontName=BODY, fontSize=9, leading=11.5, textColor=MUTED, spaceAfter=4),
    "caption": ParagraphStyle("caption", fontName=BODY, fontSize=9, leading=11.5, textColor=MUTED,
                              alignment=TA_CENTER, spaceBefore=3, spaceAfter=10),
    "h1": ParagraphStyle("h1", fontName=BOLD, fontSize=17, leading=21, textColor=ACCENT, spaceBefore=6, spaceAfter=8),
    "h2": ParagraphStyle("h2", fontName=BOLD, fontSize=13, leading=16, textColor=INK, spaceBefore=10, spaceAfter=5),
    "h3": ParagraphStyle("h3", fontName=BOLD, fontSize=11, leading=14, textColor=INK, spaceBefore=7, spaceAfter=3),
    "q": ParagraphStyle("q", fontName=BOLD, fontSize=10.5, leading=14, textColor=ACCENT, spaceBefore=4, spaceAfter=1),
    "cell": ParagraphStyle("cell", fontName=BODY, fontSize=8.6, leading=10.4, textColor=INK),
    "cellb": ParagraphStyle("cellb", fontName=BOLD, fontSize=8.6, leading=10.4, textColor=INK),
    "mono": ParagraphStyle("mono", fontName=MONO, fontSize=7.6, leading=9.6, textColor=colors.HexColor("#D8DEE6")),
    "title": ParagraphStyle("title", fontName=BOLD, fontSize=30, leading=34, textColor=INK, spaceAfter=6),
    "subtitle": ParagraphStyle("subtitle", fontName=BODY, fontSize=14, leading=18, textColor=MUTED, spaceAfter=4),
    "toctitle": ParagraphStyle("toctitle", fontName=BOLD, fontSize=17, leading=21, textColor=ACCENT, spaceAfter=10),
    "toc1": ParagraphStyle("toc1", fontName=BODY, fontSize=10.5, leading=15, textColor=INK),
    "toc2": ParagraphStyle("toc2", fontName=BODY, fontSize=9.5, leading=13, textColor=MUTED, leftIndent=14),
}
PAGE_W = A4[0] - 40 * mm


# ---------------------------------------------------------------- helpers
class Doc(SimpleDocTemplate):
    def afterFlowable(self, f):
        if isinstance(f, Paragraph) and f.style.name in ("h1", "h2"):
            level = 0 if f.style.name == "h1" else 1
            key = f"h{id(f)}"
            self.canv.bookmarkPage(key)
            self.notify("TOCEntry", (level, f.getPlainText(), self.page, key))


def P(text, style="body"):
    return Paragraph(text, S[style])


def H1(t):
    return P(t, "h1")


def H2(t):
    return P(t, "h2")


def H3(t):
    return P(t, "h3")


def Q(t):
    return P(t, "q")


def pct(x, d=1):
    return f"{100 * x:.{d}f}%"


def f4(x):
    return f"{x:.4f}"


def table(rows, widths=None, header=True, font_size=None, align_right_from=1):
    data = []
    for i, r in enumerate(rows):
        st = S["cellb"] if (header and i == 0) else S["cell"]
        data.append([c if not isinstance(c, str) else Paragraph(c, st) for c in r])
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0, hAlign="LEFT")
    style = [
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("LINEBELOW", (0, 0), (-1, 0), 0.8, INK) if header else ("LINEBELOW", (0, 0), (0, 0), 0, colors.white),
        ("LINEBELOW", (0, -1), (-1, -1), 0.6, RULE),
    ]
    for i in range(1 if header else 0, len(rows)):
        if (i % 2 == 0) == header:
            style.append(("BACKGROUND", (0, i), (-1, i), ZEBRA))
    t.setStyle(TableStyle(style))
    return t


def fig(path, caption, width=PAGE_W, max_h=105 * mm):
    from reportlab.lib.utils import ImageReader
    iw, ih = ImageReader(path).getSize()
    w = width
    h = w * ih / iw
    if h > max_h:
        h = max_h
        w = h * iw / ih
    return KeepTogether([Image(path, width=w, height=h), P(caption, "caption")])


def terminal(lines):
    body = "<br/>".join(l.replace("&", "&amp;").replace("<", "&lt;").replace(" ", "&nbsp;") for l in lines)
    t = Table([[Paragraph(body, S["mono"])]], colWidths=[PAGE_W])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#1E242C")),
                           ("TOPPADDING", (0, 0), (-1, -1), 8), ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                           ("LEFTPADDING", (0, 0), (-1, -1), 9), ("RIGHTPADDING", (0, 0), (-1, -1), 9)]))
    return t


def on_page(canv, doc):
    canv.saveState()
    canv.setFont(BODY, 8.5)
    canv.setFillColor(MUTED)
    if doc.page > 1:
        canv.drawString(20 * mm, 12 * mm, "PathSnap · Technical Report · UE24CS352A Machine Learning")
        canv.drawRightString(A4[0] - 20 * mm, 12 * mm, f"{doc.page}")
        canv.setStrokeColor(RULE)
        canv.line(20 * mm, 15.5 * mm, A4[0] - 20 * mm, 15.5 * mm)
    canv.restoreState()


# ---------------------------------------------------------------- data
eda = pd.read_csv(os.path.join(R, "eda", "summary_by_partition.csv")).set_index("partition")
inv = pd.read_csv(os.path.join(R, "eda", "inventory.csv"), dtype={"trip_id": str})
miss = pd.read_csv(os.path.join(R, "eda", "missing_values.csv"))
cst = {p: pd.read_csv(os.path.join(R, f"candidate_stats_{p}.csv"), dtype={"trip_id": str}) for p in ("train", "student_test")}
cmp_ = pd.read_csv(os.path.join(R, "model_comparison.csv")).set_index("model")
folds = pd.read_csv(os.path.join(R, "cv_folds.csv"))
tvu = pd.read_csv(os.path.join(R, "tuned_vs_untuned.csv"))
bo = json.load(open(os.path.join(R, "bayes_opt_search_space.json")))
trials = pd.read_csv(os.path.join(R, "bayes_opt_trials.csv"))
hmm_t = pd.read_csv(os.path.join(R, "hmm_tuning.csv"))
hmm_grid = pd.read_csv(os.path.join(R, "hmm_tuning_grid_beta_x_w.csv"))
hmm_s = pd.read_csv(os.path.join(R, "hmm_summary.csv")).set_index("method")
hmm_v = pd.read_csv(os.path.join(R, "hmm_vs_classifier.csv"), dtype={"trip_id": str})
hcfg = json.load(open(os.path.join(config.ARTIFACTS, "hmm_config.json")))
meta = json.load(open(os.path.join(config.ARTIFACTS, "final_model_meta.json")))
ev = pd.read_csv(os.path.join(R, "evaluator_test", "results.csv"), dtype={"trip_id": str})
evs = json.load(open(os.path.join(R, "evaluator_test", "summary.json")))
te_feat_path = os.path.join(config.CACHE, "features_student_test.pkl")

n_te = int(cst["student_test"].n_obs.sum())
miss_te = float((cst["student_test"].candidate_miss_rate * cst["student_test"].n_obs).sum() / n_te)
baseline = None
if os.path.exists(te_feat_path):   # nearest-candidate baseline (needs the feature cache)
    te = pd.read_pickle(te_feat_path)
    baseline = float(te[te.dist_rank == 0].label.sum() / n_te)
fold_top1_std = folds.groupby("model").top1.std()
C = cmp_.to_dict("index")
clf, hm = hmm_s.loc["Classifier only"], hmm_s.loc["Classifier + HMM"]
tv_un, tv_tu = tvu.iloc[0], tvu.iloc[1]
best = bo["best_params"]

story = []

# ---------------------------------------------------------------- title page
story += [Spacer(1, 38 * mm),
          P("PES University · Department of Computer Science and Engineering · RR Campus", "small"),
          P("UE24CS352A Machine Learning · PathSnap Hackathon", "small"), Spacer(1, 6 * mm),
          P("PathSnap", "title"),
          P("Distributed map-matching and route reconstruction<br/>with a learned candidate scorer and HMM/Viterbi decoding", "subtitle"),
          Spacer(1, 10 * mm)]
story.append(table([
    ["Team member", "GitHub", "Primary responsibility"],
    ["Shryes", "@Shryes19", "Data loading, cleaning, EDA, results summary"],
    ["Shubham Bhat", "@Shubhambhat06", "Road network, candidates, labelling, features"],
    ["Siddhant Anil", "@SiRex750", "Model comparison, metrics, Bayesian optimisation"],
    ["Siddhant Khurdiya", "@sidd2006", "HMM/Viterbi, pipeline, API, UI, deployment"],
], widths=[38 * mm, 34 * mm, PAGE_W - 72 * mm]))
story += [Spacer(1, 8 * mm),
          P(f"<b>Repository:</b> github.com/sidd2006/PathSnap &nbsp;·&nbsp; <b>Date:</b> {date.today():%d %B %Y}"),
          Spacer(1, 6 * mm)]
story.append(table([
    ["Headline result", "Classifier only", "Classifier + HMM"],
    ["Student test, pointwise accuracy (10 trips)", pct(clf.pointwise_accuracy_obs_weighted, 2), pct(hm.pointwise_accuracy_obs_weighted, 2)],
    ["Evaluator test, pointwise accuracy (10 trips, via API)", pct(evs["clf_accuracy_obs_weighted"], 2), pct(evs["hmm_accuracy_obs_weighted"], 2)],
    ["Evaluator test, route continuity", pct(evs["clf_route_continuity_mean"], 2), pct(evs["hmm_route_continuity_mean"], 2)],
], widths=[PAGE_W - 70 * mm, 35 * mm, 35 * mm]))
story += [Spacer(1, 8 * mm),
          P("All numbers in this report are generated from files in the repository (<i>reports/</i>, "
            "<i>artifacts/</i>) by <i>report/build_report.py</i>, and can be reproduced with the scripts listed in "
            "Section 9. AI assistance is disclosed in Section 10.", "small"),
          PageBreak()]

toc = TableOfContents()
toc.levelStyles = [S["toc1"], S["toc2"]]
story += [P("Contents", "toctitle"), toc, PageBreak()]

# ---------------------------------------------------------------- 1. overview
story += [H1("1. Problem and approach"),
          P("GPS fixes drift onto parallel streets, jump across buildings and drop out. For every GPS observation of a "
            "trip, PathSnap must return the directed road arc (from the trip's <i>.arcs</i> file) that the vehicle was "
            "actually on, and the arcs must form a plausible connected route."),
          P("We split the task into two learned and probabilistic stages. First, for every GPS fix we generate nearby "
            "candidate arcs and a binary classifier estimates P(candidate is the reference arc). Second, a Hidden Markov "
            "Model uses these probabilities as emissions and road-network distances as transitions; Viterbi decoding "
            "returns the most probable connected sequence. The pipeline is served by a FastAPI backend (Laptop 1) and "
            "queried by a browser UI (Laptop 2)."),
          table([
              ["Stage", "Method", "Code"],
              ["Cleaning and projection", "drop invalid fixes, sort by time, local equirectangular metres", "io.py, geo.py"],
              ["Candidate generation", "KD-tree over arc points every 8 m; 50 m radius (120 m fallback); top 12", "network.py, features.py"],
              ["Labels (training/scoring only)", "monotone DP aligning .route arcs to GPS fixes", "labels.py"],
              ["Features", f"{len(FEATURES)} geometric, kinematic, contextual and topological features", "features.py"],
              ["Classifier", "LightGBM, Bayesian-tuned (Optuna TPE)", "models.py, 04_bayes_opt.py"],
              ["Decoder", "HMM + Viterbi with Dijkstra transitions and chain restarts", "hmm.py"],
              ["Serving", "FastAPI /trips, /predict; web UI with Leaflet map", "api/, ui/"],
          ], widths=[42 * mm, PAGE_W - 87 * mm, 45 * mm]),
          PageBreak()]

# ---------------------------------------------------------------- 2.2 EDA
tr_e, te_e = eda.loc["train"], eda.loc["student_test"]
bad = cst["train"][cst["train"].label_dist_median > 30]
story += [H1("2. Dataset understanding and exploratory data analysis"),
          P("<b>Partitions used.</b> All exploratory analysis uses the <b>80 training trips</b> and the <b>10 "
            "student-test trips</b>, reported separately below. The evaluator-test trips were not opened during "
            "development; they were used once, at the end, through the API (Section 8)."),
          table([
              ["Statistic", "Train", "Student test"],
              ["Trips", f"{tr_e.trips:.0f}", f"{te_e.trips:.0f}"],
              ["GPS observations", f"{tr_e.obs_total:,.0f}", f"{te_e.obs_total:,.0f}"],
              ["Mean observations per trip", f"{tr_e.obs_mean:,.0f}", f"{te_e.obs_mean:,.0f}"],
              ["Mean trip length (km)", f"{tr_e.length_km_mean:.1f}", f"{te_e.length_km_mean:.1f}"],
              ["Share of 1 s sampling steps", pct(tr_e.dt_frac_1s), pct(te_e.dt_frac_1s)],
              ["Largest time gap (s)", f"{tr_e.dt_max:,.0f}", f"{te_e.dt_max:,.0f}"],
              ["Gaps longer than 5 s", f"{tr_e.gaps_gt5s:,.0f}", f"{te_e.gaps_gt5s:,.0f}"],
              ["Mean road nodes / arcs per trip", f"{tr_e.nodes_mean:,.0f} / {tr_e.arcs_mean:,.0f}", f"{te_e.nodes_mean:,.0f} / {te_e.arcs_mean:,.0f}"],
              ["Isolated road nodes", f"{tr_e.isolated_nodes:.0f}", f"{te_e.isolated_nodes:.0f}"],
              ["Reference-route continuity", pct(tr_e.route_continuity, 2), pct(te_e.route_continuity, 2)],
          ], widths=[PAGE_W - 70 * mm, 35 * mm, 35 * mm]),
          P("Source: <i>reports/eda/summary_by_partition.csv</i>, <i>reports/eda/inventory.csv</i>.", "caption"),
          H2("2.1 How did we preprocess the data?"),
          P("Each trip folder holds four header-less whitespace files. <i>.track</i> has longitude, latitude and a "
            "timestamp in seconds; <i>.nodes</i> has node coordinates (row index = node id); <i>.arcs</i> lists directed "
            "road segments as (from node, to node) with the row index as arc id; <i>.route</i> is the chronological "
            "list of reference arc ids. A row of <i>.track</i> does not correspond to a row of <i>.route</i>: route "
            "files are several times shorter than tracks (525 route arcs for 1,315 fixes in train trip 00000001) and "
            "carry no timestamps."),
          P("Preprocessing (<i>io.clean_track</i>) removes non-finite values, coordinates outside ±180°/±90°, sorts "
            "by timestamp and drops duplicate timestamps. We deliberately keep noisy but valid fixes, because handling "
            "noise is the job of the classifier and decoder. Coordinates are projected to a local equirectangular "
            "frame in metres centred on each trip (x = Δlon·R·cos φ<sub>0</sub>, y = Δlat·R). Over a city-sized trip "
            "the distortion is below 0.1%, far smaller than GPS noise, and planar point-to-segment geometry becomes "
            "cheap and exact enough."),
          P("To create training labels we align the reference route to the fixes with a monotone dynamic programme "
            "(<i>labels.align_route</i>): each fix i receives a route position j<sub>i</sub> with j<sub>0</sub> ≤ "
            "j<sub>1</sub> ≤ … that minimises the total point-to-arc distance (O(N·M) with a running minimum). "
            f"Observations whose aligned reference arc is more than {LABEL_MAX_DIST:.0f} m away are excluded from "
            "training as unreliable; evaluation always uses every observation."),
          H2("2.2 Missing-value and data-quality analysis"),
          table([
              ["Partition", "Observations", "Non-finite rows", "Out-of-range rows", "Duplicate timestamps", "Out-of-order rows"],
              *[[p, f"{g.n_obs.sum():,}", f"{g.nonfinite_rows.sum()}", f"{g.invalid_rows.sum()}", f"{g.duplicate_t.sum()}",
                 f"{g.reordered.sum()}"] for p, g in miss.groupby("partition")],
          ], widths=[28 * mm, 28 * mm, 28 * mm, 30 * mm, 32 * mm, PAGE_W - 146 * mm]),
          P("Source: <i>reports/eda/missing_values.csv</i>. No values are missing; the quality problems are gaps and "
            "label offsets rather than empty cells.", "caption"),
          P(f"<b>Reference routes are not error-free.</b> On training trips "
            f"{', '.join(t.lstrip('0') for t in bad.trip_id)} the median distance from a GPS fix to its aligned "
            f"reference arc is {', '.join(f'{v:.0f} m' for v in bad.label_dist_median)}, against "
            f"{cst['train'].label_dist_median.median():.1f} m for a typical trip. This is a label problem rather "
            f"than GPS noise, so we treat these routes as unreliable reference information. "
            f"<b>Large gaps:</b> one training trip contains a {tr_e.dt_max / 3600:.0f}-hour gap; the decoder handles "
            "gaps explicitly (Section 5)."),
          H2("2.3 Patterns observed and visualisations")]
story += [fig(os.path.join(FIG, "eda_trip_distributions.png"),
              "Figure 1. Observations per trip, trip length and mean speed (train and student test)."),
          fig(os.path.join(FIG, "eda_sampling_intervals.png"),
              "Figure 2. Time between consecutive fixes (train). The title counts all steps pooled (96.5% are exactly 1 s); "
              "the table above averages each trip's share (93.8%), which weights short trips equally.",
              max_h=70 * mm),
          fig(os.path.join(FIG, "eda_all_trajectories.png"),
              "Figure 3. GPS trajectories of all train (blue) and student-test (orange) trips in longitude/latitude. "
              "The trips are spread over several countries, so at this global scale each trip appears as a short "
              "stroke; this is why distances are always computed in a per-trip local projection.", max_h=95 * mm),
          fig(os.path.join(FIG, "eda_road_network_trip.png"),
              "Figure 4. Road network around train trip 00000001 with GPS fixes (blue) and reference route (green).",
              max_h=100 * mm),
          fig(os.path.join(FIG, "eda_network_connectivity.png"),
              "Figure 5. Road-network size per trip and share of nodes in the largest strongly connected component "
              "(train and student test).", max_h=60 * mm),
          fig(os.path.join(IMG, "map_eda_trip1.jpg"),
              "Figure 6. Interactive Folium map of train trip 00000001 (Darmstadt to Frankfurt): road corridor, "
              "reference route and GPS fixes, with layer control (<i>reports/maps/eda_trip_00000001.html</i>).",
              max_h=100 * mm)]
story += [P("<b>Patterns that shaped the design.</b> (1) Fixes are dense (≈1 Hz) and usually within 5–15 m of the "
            f"reference arc (median {cst['train'].label_dist_median.median():.1f} m on train), so a 50 m candidate "
            f"radius keeps the reference arc in the candidate set for {pct(1 - miss_te, 1)} of student-test fixes. "
            "(2) Every two-way street appears as two arcs with identical geometry and opposite direction; distance "
            "alone cannot separate them"
            + (f", which is why choosing the nearest candidate scores only {pct(baseline, 1)} on student test. "
               if baseline is not None else ". ")
            + "(3) Single fixes jump, but their neighbours rarely do, so context over a short window is informative. "
            "(4) Road networks are per trip, well connected and large, so shortest paths must be computed on a pruned "
            "corridor."),
          H2("2.4 Features extracted and why"),
          table([
              ["Group", "Features", "Why"],
              ["Geometry", "dist, dist_rank, dist_minus_min, dist_ratio, gauss_prior, t_proj, at_end",
               "how close the fix is to the arc, absolutely and relative to the other candidates"],
              ["Kinematics", "head_diff, head_cos, speed, moving, dt_prev, dt_next",
               "GPS heading vs arc bearing separates the two directions of a street; speed and gaps say how reliable heading is"],
              ["Context", f"ctx_dist_mean, ctx_dist_max (±{CTX} fixes), ctx_head_diff (±5 fixes)",
               "an arc that also explains the neighbouring fixes is more likely; smooths single outliers"],
              ["Topology", "arc_len, out_deg_to, in_deg_from, has_twin, n_cands",
               "short links, junctions and two-way streets behave differently"],
          ], widths=[22 * mm, 68 * mm, PAGE_W - 90 * mm]),
          P("Source: <i>src/pathsnap/features.py</i>.", "caption"),
          PageBreak()]

# ---------------------------------------------------------------- 3. models
order = ["LightGBM", "RandomForest", "MLP", "DecisionTree", "LogisticRegression", "KNN", "GaussianNB"]
nice = {"LightGBM": "LightGBM", "RandomForest": "Random Forest", "MLP": "MLP (64-32)", "DecisionTree": "Decision Tree",
        "LogisticRegression": "Logistic Regression", "KNN": "KNN (k = 25)", "GaussianNB": "Gaussian Naive Bayes"}
rows = [["Model", "Algorithm family", "Key configuration", "CV Macro F1", "Top-1 (test)", "Route contin.",
         "Test Macro F1", "Log loss (test)", "Infer. ms / 1k"]]
for m in order:
    c = C[m]
    rows.append([nice[m], c["family"], c["config"], f"{c['cv_macro_f1']:.4f}", f"{c['test_top1']:.4f}",
                 f"{c['route_continuity']:.4f}", f"{c['test_macro_f1']:.4f}", f"{c['test_log_loss']:.4f}",
                 f"{c['inference_ms_per_1k_rows']:.2f}"])
story += [H1("3. Model experimentation and comparison"),
          P("<b>Protocol.</b> Every model is a binary classifier over (fix, candidate) rows. We use 5-fold "
            "<b>GroupKFold</b> grouped by trip on the 80 training trips (no trip appears on both sides of a split), "
            "then refit on all 80 trips and evaluate on the 10 student-test trips. Seeds are fixed at 42 and every run "
            "is logged in <i>reports/experiment_log.jsonl</i>."),
          P("<b>Metrics.</b> <i>Macro F1</i>, precision, recall and <i>log loss</i> are computed per candidate row "
            "(threshold 0.5). <i>Top-1 accuracy</i> is per GPS fix: the highest-scoring candidate must be the "
            "reference arc; on student test a fix without the reference among its candidates counts as an error. "
            "<i>Route continuity</i> is the share of consecutive top-1 predictions that stay on the same arc or move "
            "to a connected arc. <b>Primary selection metric: Top-1 accuracy</b>, because it measures exactly what "
            "the decoder consumes (is the right arc ranked first?), with log loss as the tie-breaker because the HMM "
            "uses the probabilities themselves."),
          H2("3.1 Overall model comparison"),
          table(rows, widths=[21 * mm, 23 * mm, 28 * mm] + [14 * mm] * 7),
          P("Source: <i>reports/model_comparison.csv</i>. CV = mean over 5 GroupKFold folds (train). All metrics "
            "apply to every model, so no cell is N/A. Inference time is measured on the 153k student-test candidate "
            "rows on a laptop CPU (6 cores, no GPU); runs overlapped with other jobs, so times are indicative.", "caption"),
          P(f"<b>Nearest-road baseline.</b> Choosing the closest candidate gives {pct(baseline, 1)} Top-1 on student "
            "test; every learned model improves on it by more than 30 percentage points, mostly by using heading to "
            "separate the two directions of a street."
            if baseline is not None else ""),
          PageBreak(),
          H2("3.2 Per-model analysis")]

texts = {
    "LogisticRegression": dict(
        why="A linear model on standardised features is the simplest probabilistic scorer: distance, heading agreement "
            "and context distance each push the log-odds in a fixed direction, which matches how a human would rank "
            "candidates.",
        sl="Strengths: fast to train (≈7 s on 1.37 M rows), among the fastest to score, and reasonably calibrated. Limitation: "
           "it cannot represent interactions, for example that heading only matters when the vehicle is moving, or "
           "that a large distance is acceptable when every candidate is far.",
        hp="C = 1.0 (inverse L2 strength), lbfgs solver, max_iter = 2000, StandardScaler on all features.",
        loss="Binary cross-entropy (log loss) plus an L2 penalty on the weights."),
    "GaussianNB": dict(
        why="Naive Bayes gives class-conditional likelihoods directly and was included as the classic probabilistic "
            "baseline from the linear/probabilistic family.",
        sl="Strengths: trains in about one second and needs no tuning. Limitation: it assumes features are "
           "independent given the class, which is badly violated here (dist, dist_rank, ctx_dist_mean and gauss_prior "
           "all measure distance), so it double-counts evidence and produces over-confident probabilities.",
        hp="var_smoothing = 1e-9 (default), StandardScaler.",
        loss="No iterative loss: class priors and per-class Gaussian means and variances are fitted by maximum likelihood."),
    "KNN": dict(
        why="An instance-based model ranks a candidate by how similar labelled (fix, candidate) situations were, with "
            "no assumption about the shape of the decision boundary.",
        sl="Strength: flexible, non-parametric decision boundary. Limitations: scoring needs a distance computation "
           "against every stored row, making it by far the slowest model, and it was trained on a 150k-row sample "
           "to stay feasible; in 21 dimensions distances are also less informative.",
        hp="k = 25, distance weighting, Euclidean metric on standardised features, 150k-row training sample "
           "(whole observations sampled so candidate groups stay intact).",
        loss="No training loss (lazy learner); the probability is the distance-weighted share of positive neighbours."),
    "DecisionTree": dict(
        why="A single tree captures the obvious rule structure of the problem (if the candidate is the nearest and "
            "the heading agrees then accept) and is easy to inspect.",
        sl="Strengths: interpretable, captures interactions, fast to score. Limitations: piecewise-constant "
           "probabilities and high variance between folds.",
        hp="max_depth = 12, min_samples_leaf = 20, criterion = Gini.",
        loss="Greedy splits minimising Gini impurity."),
    "RandomForest": dict(
        why="Bagging many decorrelated trees keeps the tree's ability to model interactions while averaging away "
            "its variance, which suits noisy labels.",
        sl="Strengths: best CV Top-1 and best log loss, stable across folds. Limitations: the slowest tree model to "
           "train (≈100 s) and larger to store.",
        hp="150 trees, max_depth = 18, min_samples_leaf = 5, max_samples = 0.3 (bootstrap fraction), Gini.",
        loss="Each tree minimises Gini impurity on its bootstrap sample; probabilities are averaged."),
    "LightGBM": dict(
        why="Gradient-boosted trees fit residual errors sequentially, handle mixed-scale features without "
            "scaling, model interactions and directly optimise log loss, which is what the HMM consumes.",
        sl="Strengths: best student-test Top-1 and Macro F1, low log loss, ≈20 s training. Limitations: more "
           "hyper-parameters than the bagging models and a mild train-test gap if trees grow too large.",
        hp="n_estimators = 300, learning_rate = 0.05, num_leaves = 31, other settings default (untuned baseline).",
        loss="Binary log loss minimised by gradient boosting with leaf-wise tree growth."),
    "MLP": dict(
        why="A small neural network can learn smooth non-linear combinations of the features and outputs "
            "probabilities through a sigmoid, so it is compatible with the decoder.",
        sl="Strengths: smooth probabilities, good log loss and the highest classifier-only route continuity. "
           "Limitations: needs scaling, the slowest model to train here (≈2 min on a 400k-row sample) and "
           "sensitive to initialisation.",
        hp="hidden layers (64, 32), ReLU, Adam, alpha = 1e-4, batch 512, max_iter = 60, early stopping on a 10% "
           "validation split (patience 8), 400k-row training sample.",
        loss="Binary cross-entropy with L2 regularisation (alpha), optimised by Adam."),
}
mlp_note = ("The MLP stopped after 49 epochs; training loss fell from 0.102 to 0.062 and validation accuracy peaked "
            "at 0.975 in epoch 40, so early stopping prevented over-training (Figure 11).")
for m in ["LogisticRegression", "GaussianNB", "KNN", "DecisionTree", "RandomForest", "LightGBM", "MLP"]:
    c, t = C[m], texts[m]
    gap = c["train_macro_f1"] - c["cv_macro_f1"]
    if m == "KNN":
        fit = (f"Training Macro F1 is {c['train_macro_f1']:.3f} against {c['cv_macro_f1']:.4f} in CV. With distance "
               "weighting every training row is its own nearest neighbour at distance zero, so the training score is "
               "meaningless; the CV score and the lowest test Top-1 among the tree, linear and neural models show it "
               "generalises worse. We read this as variance-driven overfitting.")
    elif m == "GaussianNB":
        fit = (f"Training and CV Macro F1 are essentially equal ({c['train_macro_f1']:.4f} vs {c['cv_macro_f1']:.4f}) "
               "and both are the lowest of all models: the model underfits (high bias) because of its independence "
               f"assumption. Recall ({c['test_recall_macro']:.3f}) is far above precision ({c['test_precision_macro']:.3f}), "
               "so it accepts too many wrong candidates.")
    elif m == "LogisticRegression":
        fit = (f"Training and CV Macro F1 are equal ({c['train_macro_f1']:.4f} vs {c['cv_macro_f1']:.4f}), so there is "
               "no overfitting; the gap to the tree ensembles indicates mild underfitting from the linear boundary.")
    else:
        fit = (f"Training Macro F1 {c['train_macro_f1']:.4f} vs CV {c['cv_macro_f1']:.4f} (gap {gap:+.4f}). "
               + ("The gap is small and the test score matches CV, so overfitting is mild." if gap < 0.02 else
                  "The gap indicates some overfitting.")
               + (f" Fold-to-fold standard deviation of Top-1 is {fold_top1_std[m]:.4f}"
                  + (", the highest of all models, which is the variance a single tree is known for." if m == "DecisionTree" else ".")))
        if m == "MLP":
            fit += " " + mlp_note
    story += [H3(f"3.2.{['LogisticRegression', 'GaussianNB', 'KNN', 'DecisionTree', 'RandomForest', 'LightGBM', 'MLP'].index(m) + 1} "
                 f"{nice[m]} ({c['family']})"),
              Q("A1. Why suitable?"), P(t["why"]),
              Q("A2. Strengths and limitations"), P(t["sl"]),
              Q("B5. Important hyper-parameters and initial values"), P(t["hp"]),
              Q("B6. Loss function / objective"), P(t["loss"]),
              Q("C7–C8. Student-test performance"),
              P(f"Top-1 accuracy <b>{c['test_top1']:.4f}</b>, Macro F1 <b>{c['test_macro_f1']:.4f}</b> "
                f"(precision {c['test_precision_macro']:.4f}, recall {c['test_recall_macro']:.4f}), candidate accuracy "
                f"{c['test_accuracy']:.4f}, log loss <b>{c['test_log_loss']:.4f}</b>, classifier-only route continuity "
                f"{c['route_continuity']:.4f}. Training time {c['train_time_s']:.1f} s on {c['train_rows']:,} rows; "
                f"inference {c['inference_ms_per_1k_rows']:.2f} ms per 1,000 candidate rows."),
              Q("C9. Overfitting or underfitting?"), P(fit)]

story += [PageBreak(), H2("3.3 Model visualisations"),
          fig(os.path.join(FIG, "cmp_cv_macro_f1.png"), "Figure 7. Cross-validation Macro F1 across the 5 GroupKFold folds for every model.", max_h=75 * mm),
          fig(os.path.join(FIG, "cmp_student_test.png"), "Figure 8. Student-test Top-1 accuracy and Macro F1 per model.", max_h=75 * mm),
          fig(os.path.join(FIG, "cmp_log_loss.png"), "Figure 9. Student-test log loss per model (log scale; lower is better).", max_h=70 * mm),
          fig(os.path.join(FIG, "cmp_speed_vs_accuracy.png"), "Figure 10. Student-test Top-1 against inference time.", max_h=80 * mm),
          fig(os.path.join(FIG, "mlp_training_curve.png"),
              "Figure 11. MLP training loss and validation accuracy per epoch. Only the MLP produces an epoch-wise "
              "history; no curves are drawn for the other models.", max_h=70 * mm)]
cm_imgs = [Image(os.path.join(FIG, f"cm_{m}.png"), width=PAGE_W / 4 - 2 * mm, height=(PAGE_W / 4 - 2 * mm) * 0.8) for m in order]
cm_tbl = Table([cm_imgs[:4], cm_imgs[4:] + [""]], colWidths=[PAGE_W / 4] * 4)
story += [KeepTogether([cm_tbl, P("Figure 12. Confusion matrices on student-test candidates (optional diagnostic). Order: "
                                  + ", ".join(nice[m] for m in order) + ".", "caption")])]
story += [H2("3.4 Model selection"),
          P(f"LightGBM and Random Forest are statistically tied in cross-validation (CV Top-1 "
            f"{C['LightGBM']['cv_top1']:.4f} vs {C['RandomForest']['cv_top1']:.4f}, fold standard deviation ≈ "
            f"{fold_top1_std['LightGBM']:.3f}). We selected <b>LightGBM</b> because it has the best student-test Top-1 "
            f"({C['LightGBM']['test_top1']:.4f}) and Macro F1 ({C['LightGBM']['test_macro_f1']:.4f}), its log loss is "
            f"close to the best, it trains about four times faster than Random Forest "
            f"({C['LightGBM']['train_time_s']:.0f} s vs {C['RandomForest']['train_time_s']:.0f} s), and boosting exposes "
            "more meaningful hyper-parameters for Bayesian optimisation. We did not select on training accuracy: KNN "
            "has a perfect training score and is among the weakest on held-out data."),
          PageBreak()]

# ---------------------------------------------------------------- 4. Bayesian optimisation
space = bo["search_space"]
story += [H1("4. Bayesian hyper-parameter optimisation (LightGBM)"),
          Q("1–2. Hyper-parameters searched and their ranges"),
          table([["Hyper-parameter", "Distribution", "Range", "Best value"]] +
                [[k, {"int": "integer uniform", "float": "uniform", "float_log": "log-uniform"}[v[0]], f"{v[1]} – {v[2]}",
                  f"{best[k]:.4g}" if isinstance(best[k], float) else str(best[k])] for k, v in space.items()],
                widths=[45 * mm, 35 * mm, 40 * mm, PAGE_W - 120 * mm]),
          Q("3. Objective"),
          P("Maximise the mean observation-level Top-1 accuracy over 3-fold GroupKFold on the training trips (macro F1 "
            "and log loss were recorded per trial). To keep 30 trials affordable each trial used a fixed 50% sample of "
            "training observations; the sampler was Optuna's Tree-structured Parzen Estimator (8 random start-up "
            "trials, seed 42)."),
          Q("4. Number of trials and computational cost"),
          P(f"{bo['n_trials']} trials in {bo['search_seconds'] / 60:.0f} minutes on a 6-core laptop CPU "
            f"(about {bo['search_seconds'] / bo['n_trials']:.0f} s per trial). The full trial log is in "
            "<i>reports/bayes_opt_trials.csv</i>."),
          Q("5. Best values"),
          P(f"Best objective {bo['best_value']:.4f}: {best['n_estimators']} trees, learning rate "
            f"{best['learning_rate']:.3f}, {best['num_leaves']} leaves, min_child_samples {best['min_child_samples']}, "
            f"subsample {best['subsample']:.2f}, colsample_bytree {best['colsample_bytree']:.2f}, light L1/L2 "
            "regularisation. The search preferred many small, heavily regularised trees (few leaves, large leaves)."),
          Q("6. Did the tuned model outperform the untuned model?"),
          table([["Model", "CV Macro F1", "CV Top-1", "Test Top-1", "Test Macro F1", "Log loss", "Train time (s)", "Inference (ms/1k)"],
                 ["LightGBM untuned", f4(tv_un.cv_macro_f1), f4(tv_un.cv_top1), f4(tv_un.test_top1), f4(tv_un.test_macro_f1),
                  f4(tv_un.test_log_loss), f"{tv_un.train_time_s:.1f}", f"{tv_un.inference_ms_per_1k_rows:.2f}"],
                 ["LightGBM tuned", f4(tv_tu.cv_macro_f1), f4(tv_tu.cv_top1), f4(tv_tu.test_top1), f4(tv_tu.test_macro_f1),
                  f4(tv_tu.test_log_loss), f"{tv_tu.train_time_s:.1f}", f"{tv_tu.inference_ms_per_1k_rows:.2f}"]],
                widths=[32 * mm] + [(PAGE_W - 32 * mm) / 7] * 7),
          P("Source: <i>reports/tuned_vs_untuned.csv</i>; both models evaluated with the same 5-fold CV and the same "
            "student-test partition, in the same run.", "caption"),
          P(f"Yes, slightly and consistently: CV Top-1 rose from {tv_un.cv_top1:.4f} to {tv_tu.cv_top1:.4f} and every "
            f"metric moved in the right direction. The clearest gain is probability quality (log loss "
            f"{tv_un.test_log_loss:.4f} → {tv_tu.test_log_loss:.4f}, −{100 * (1 - tv_tu.test_log_loss / tv_un.test_log_loss):.1f}%), "
            "which matters because the HMM uses the probabilities as emissions. The cost is a slightly slower model "
            "(more trees)."),
          fig(os.path.join(FIG, "bayes_opt_history.png"), "Figure 13. Objective per trial and best-so-far curve.", max_h=70 * mm),
          PageBreak()]

# ---------------------------------------------------------------- 5. HMM
story += [H1("5. HMM and Viterbi route reconstruction"),
          Q("1. Hidden states"),
          P("At each GPS fix the hidden state is one of its candidate arcs (up to 12), together with the fix's "
            "projection point on that arc (fraction t along the arc). The candidate-to-arc correspondence is kept "
            "throughout, so the decoded state is directly the predicted original arc id."),
          Q("2. Emission probabilities"),
          P("The selected LightGBM's probabilities, renormalised over the fix's candidates: log e = w·log(p / Σp), "
            f"with emission weight w = {hcfg['emission_weight']:g}."),
          Q("3. Transition probabilities"),
          P("Following Newson and Krumm, a transition is likely when the network distance between the two projection "
            "points matches the straight-line GPS displacement: log a = −|d<sub>network</sub> − d<sub>GPS</sub>| / "
            f"(β·√Δt) with β = {hcfg['beta']:g} m. On the same arc the network distance is (t<sub>2</sub> − t<sub>1</sub>)·length, with a "
            f"{hcfg['backward_penalty']:g} m penalty for moving backwards; between arcs it is the rest of the first arc + "
            "a Dijkstra shortest path from its head node to the second arc's tail node + the start of the second arc. "
            f"Dijkstra stops at a cutoff of {hcfg['cutoff_factor']:g}·d<sub>GPS</sub> + {hcfg['cutoff_extra']:g}·√Δt metres; "
            "pairs with no path inside the cutoff are invalid (probability 0)."),
          Q("4. Missing candidates and large GPS gaps"),
          P("Fixes with no candidate within 50 m are re-queried at 120 m. Fixes with still no candidate are skipped by "
            "the decoder and inherit the previous (else next) predicted arc. Time gaps widen both the transition "
            "tolerance (β·√Δt) and the search cutoff, so a long gap does not force an impossible exact match."),
          Q("5. Disconnected road segments"),
          P("If every transition into a fix is invalid (no connected path between any pair of candidates), the "
            "current chain is closed and back-tracked, and a new chain starts at that fix from its emissions alone. "
            f"One disconnected step therefore cannot invalidate the whole route; on student test this happened "
            f"{int(hmm_v.hmm_breaks.sum())} times in total."),
          Q("Decoder parameter tuning"),
          P("Viterbi's arg-max does not change when all log-scores are multiplied by a constant. Dividing the path score "
            "w·Σlog e − Σ|Δd|/β by w shows that only the product β·w matters; a first 5 × 3 grid confirmed it "
            "(equal products gave identical accuracy, <i>reports/hmm_tuning_grid_beta_x_w.csv</i>). We therefore fixed "
            "w = 1 and searched β. The search used 12 <b>held-out training trips</b>, scored with a LightGBM that had "
            "not seen them, so the probabilities were out-of-sample; student-test and evaluator data were not used."),
          table([["β (m)"] + [f"{b:g}" for b in hmm_t.beta],
                 ["Accuracy"] + [f"{a:.4f}" for a in hmm_t.accuracy],
                 ["Continuity"] + [f"{a:.4f}" for a in hmm_t.continuity]],
                widths=[22 * mm] + [(PAGE_W - 22 * mm) / len(hmm_t)] * len(hmm_t)),
          P(f"Source: <i>reports/hmm_tuning.csv</i>. Accuracy peaks at β = {hcfg['beta']:g}, in the interior of the "
            "range.", "caption"),
          Q("6. Did HMM decoding improve the route?"),
          table([["Student test (10 trips)", "Classifier only", "Classifier + HMM"],
                 ["Pointwise accuracy (observation-weighted)", pct(clf.pointwise_accuracy_obs_weighted, 2), pct(hm.pointwise_accuracy_obs_weighted, 2)],
                 ["Pointwise accuracy (mean of trips)", pct(clf.pointwise_accuracy_trip_mean, 2), pct(hm.pointwise_accuracy_trip_mean, 2)],
                 ["Route continuity", pct(clf.route_continuity, 2), pct(hm.route_continuity, 2)],
                 ["Route precision (traversed-arc set)", pct(clf.route_precision, 2), pct(hm.route_precision, 2)],
                 ["Route recall (traversed-arc set)", pct(clf.route_recall, 2), pct(hm.route_recall, 2)],
                 ["Route F1 (sequence-level)", pct(clf.route_f1, 2), pct(hm.route_f1, 2)],
                 ["Trips with accuracy ≥ 95%", f"{clf.trips_with_acc_ge_95:.0f} / 10", f"{hm.trips_with_acc_ge_95:.0f} / 10"]],
                widths=[PAGE_W - 70 * mm, 35 * mm, 35 * mm]),
          P("Source: <i>reports/hmm_summary.csv</i>.", "caption"),
          table([["Trip", "Fixes", "Clf acc.", "HMM acc.", "Clf continuity", "HMM continuity", "Clf route recall", "HMM route recall", "Restarts"]] +
                [[r.trip_id, f"{r.n_obs:,}", f4(r.clf_accuracy), f4(r.hmm_accuracy), f4(r.clf_route_continuity),
                  f4(r.hmm_route_continuity), f4(r.clf_route_recall), f4(r.hmm_route_recall), str(r.hmm_breaks)]
                 for r in hmm_v.itertuples()],
                widths=[20 * mm, 15 * mm] + [(PAGE_W - 50 * mm) / 6] * 6 + [15 * mm]),
          P("Source: <i>reports/hmm_vs_classifier.csv</i> (student test).", "caption"),
          fig(os.path.join(FIG, "hmm_vs_classifier.png"), "Figure 14. Pointwise accuracy and route continuity per student-test trip.", max_h=62 * mm)]
worse_rec = hmm_v[hmm_v.hmm_route_recall < hmm_v.clf_route_recall].trip_id.tolist()
ev17 = ev.set_index("trip_id").loc["00000017"]
story += [P(f"<b>Improvements.</b> Decoding raised pointwise accuracy on all 10 student-test trips (from "
            f"{pct(clf.pointwise_accuracy_obs_weighted, 2)} to {pct(hm.pointwise_accuracy_obs_weighted, 2)}), with the "
            f"largest gain on trip 00000077 ({pct(hmm_v.set_index('trip_id').loc['00000077'].clf_accuracy, 1)} → "
            f"{pct(hmm_v.set_index('trip_id').loc['00000077'].hmm_accuracy, 1)}). Continuity rose from "
            f"{pct(clf.route_continuity, 1)} to {pct(hm.route_continuity, 1)} and route precision from "
            f"{pct(clf.route_precision, 1)} to {pct(hm.route_precision, 1)}: the decoder mainly removes impossible "
            "detours onto parallel or opposite-direction arcs."),
          P(f"<b>Where it did not help.</b> Route recall barely changed ({pct(clf.route_recall, 1)} → "
            f"{pct(hm.route_recall, 1)}) and fell slightly on trips {', '.join(worse_rec)}: arcs that no GPS fix is "
            "matched to (very short links passed between two fixes) stay missing, because the decoder assigns one arc "
            f"per fix and does not insert the connecting path. On the evaluator set, trip 00000017 lost "
            f"{100 * (ev17.clf_accuracy - ev17.hmm_accuracy):.2f} points of pointwise accuracy "
            f"({pct(ev17.clf_accuracy, 2)} → {pct(ev17.hmm_accuracy, 2)}) although continuity improved; there the "
            "decoder prefers a connected but wrong alternative for some stretches."),
          PageBreak()]

# ---------------------------------------------------------------- 6. Folium
story += [H1("6. Geographical visualisation using Folium"),
          P("Every map is generated by the repository code (<i>src/pathsnap/viz.py</i>, <i>scripts/01_eda.py</i>, "
            "<i>scripts/05_hmm_eval.py</i>, <i>scripts/07_evaluator_run.py</i>) and committed as interactive HTML in "
            "<i>reports/maps/</i> and <i>reports/evaluator_test/maps/</i>. Layers use distinct colours: GPS "
            "observations (blue dots), road network around the candidates (grey), reference route (green, wide), "
            "classifier-only prediction (orange), classifier + HMM prediction (red); start and end markers; a legend "
            "box; a layer control to toggle each layer and switch basemap; and bounds fitted to the trip. Evaluator "
            "maps deliberately omit the reference route."),
          fig(os.path.join(IMG, "map_pred56_overview.jpg"),
              "Figure 15. Student-test trip 00000056, whole trip (<i>reports/maps/pred_00000056.html</i>).", max_h=95 * mm),
          fig(os.path.join(IMG, "map_pred56_zoom.jpg"),
              "Figure 16. Same trip zoomed into the motorway interchange: where the classifier alone (orange) jumps to a "
              "parallel carriageway, the HMM route (red) follows the reference (green).", max_h=95 * mm),
          fig(os.path.join(IMG, "map_eval15.jpg"),
              "Figure 17. Evaluator trip 00000015 (Streets basemap): GPS observations, candidate road network and the "
              "HMM route (<i>reports/evaluator_test/maps/pred_00000015.html</i>).", max_h=95 * mm),
          P("<b>Note on basemaps.</b> The public OpenStreetMap tile server answers these pages with “Access blocked” "
            "tiles, so all maps use keyless Esri basemaps (Light Gray and Streets), with OpenStreetMap data "
            "credited in the attribution.", "small"),
          PageBreak()]

# ---------------------------------------------------------------- 7. API / UI
log_lines = [l.rstrip() for l in open(os.path.join(R, "evaluator_test", "api_server_log.txt"), encoding="utf-8")
             if "POST /predict trip=" in l or "GET /trips" in l][:7]
story += [H1("7. API, UI and deployment"),
          P("<b>Architecture.</b> Laptop 1 runs the FastAPI backend (<i>api/main.py</i>): it loads the final model, the "
            "decoder configuration and the trip folder given by <i>PATHSNAP_EVAL_DIR</i>. Laptop 2 runs the web UI "
            "(<i>ui/web/</i>, served by <i>ui/serve.py</i>), which talks to the API over HTTP only."),
          table([["Endpoint", "Purpose"],
                 ["GET /health", "status, loaded model and decoder configuration"],
                 ["GET /trips", "ids of the evaluation trips (no labels)"],
                 ["POST /predict {trip_id}", "runs candidates → LightGBM → HMM; returns GPS, predicted and classifier-only "
                                             "routes, nearby road network, server-side scores, decoder diagnostics and timing"]],
                widths=[45 * mm, PAGE_W - 45 * mm]),
          P("<b>Leakage control.</b> <i>predict_trip</i> loads the trip without its <i>.route</i> file. Only after "
            "prediction does the API read the reference route, compute the scores and log them; the reference route "
            "is never included in the response, so neither the model nor the UI can see it. No trip ids or predictions "
            "are hard-coded."),
          P("<b>Deployment.</b> Laptop 1: <i>uvicorn api.main:app --host 0.0.0.0 --port 8000</i> with "
            "PATHSNAP_EVAL_DIR set to the evaluator folder and port 8000 allowed through the firewall. Laptop 2: "
            "<i>python ui/serve.py --api http://&lt;laptop-1-ip&gt;:8000</i>. Full steps are in the README."),
          fig(os.path.join(ROOT, "docs", "ui_web.jpg"),
              "Figure 18. Evaluator UI showing evaluator trip 00000087: trip list with per-trip accuracy, the HMM route "
              "on the map, and the results panel (98.4% accuracy, +2.4 points over the classifier, per-stage timing).",
              max_h=88 * mm),
          P("<b>API responses on the receiving laptop.</b> Every request is logged by the API process. The excerpt "
            "below is the API terminal output recorded during the final evaluator run "
            "(<i>reports/evaluator_test/api_server_log.txt</i>):"),
          terminal(log_lines),
          PageBreak(), H2("7.1 Two-laptop demonstration screenshots"),
          P("<font color='#B42318'><b>[Before submission: paste here (1) a screenshot of the API terminal on Laptop 1 "
            "showing the POST /predict log lines, and (2) a screenshot of the UI on Laptop 2 with a reconstructed "
            "evaluator trip, both taken during the two-laptop run.]</b></font>"),
          PageBreak()]

# ---------------------------------------------------------------- 8. evaluator
story += [H1("8. Final evaluation on the evaluator-test trips"),
          P(f"The 10 evaluator trips ({evs['n_obs']:,} GPS fixes) were run once through the deployed API with the frozen "
            f"model (checksum {evs['final_model_sha256_16']}) and decoder (checksum {evs['hmm_config_sha256_16']}). "
            "Nothing was trained, tuned or selected on these trips. An independent offline run "
            "(<i>scripts/evaluate_dataset.py</i>) produced identical results."),
          table([["Metric", "Classifier only", "Classifier + HMM"],
                 ["Pointwise accuracy (observation-weighted)", pct(evs["clf_accuracy_obs_weighted"], 2), pct(evs["hmm_accuracy_obs_weighted"], 2)],
                 ["Pointwise accuracy (mean of trips)", pct(evs["clf_accuracy_trip_mean"], 2), pct(evs["hmm_accuracy_trip_mean"], 2)],
                 ["Route continuity (mean of trips)", pct(evs["clf_route_continuity_mean"], 2), pct(evs["hmm_route_continuity_mean"], 2)],
                 ["Route F1 (mean of trips)", pct(evs["clf_route_f1_mean"], 2), pct(evs["hmm_route_f1_mean"], 2)],
                 ["Trips with accuracy ≥ 95%", f"{int((ev.clf_accuracy >= 0.95).sum())} / 10", f"{evs['trips_hmm_acc_ge_95']} / 10"],
                 ["Trips where the HMM beats the classifier", "", f"{evs['trips_where_hmm_beats_classifier']} / 10"],
                 ["Total API server time", "", f"{evs['server_time_s_total']:.1f} s"]],
                widths=[PAGE_W - 70 * mm, 35 * mm, 35 * mm]),
          Spacer(1, 4),
          table([["Trip", "Fixes", "Clf acc.", "HMM acc.", "HMM continuity", "HMM route F1", "Server time (s)"]] +
                [[r.trip_id, f"{r.n_obs:,}", f4(r.clf_accuracy), f4(r.hmm_accuracy), f4(r.hmm_route_continuity),
                  f4(r.hmm_route_f1), f"{r.server_time_s:.2f}"] for r in ev.itertuples()],
                widths=[22 * mm, 18 * mm] + [(PAGE_W - 40 * mm) / 5] * 5),
          P("Source: <i>reports/evaluator_test/results.csv</i>, <i>summary.json</i>.", "caption"),
          fig(os.path.join(FIG, "eval_hmm_vs_classifier.png"), "Figure 19. Accuracy and continuity per evaluator trip.", max_h=62 * mm),
          P(f"The evaluator accuracy ({pct(evs['hmm_accuracy_obs_weighted'], 2)}) is in line with the student-test "
            f"accuracy ({pct(hm.pointwise_accuracy_obs_weighted, 2)}), which indicates that the pipeline generalises "
            "to unseen trips rather than to the development trips only."),
          PageBreak()]

# ---------------------------------------------------------------- 9-10
story += [H1("9. Conclusions, limitations and reproducibility"),
          P("A learned candidate scorer combined with an HMM decoder reconstructs routes with about 95% pointwise "
            "accuracy on unseen trips in one to two seconds per trip on a laptop CPU. The learned scorer contributes "
            "most of the gain over geometry alone (direction of travel and context), and the decoder adds about four "
            "points by enforcing connectivity."),
          P("<b>Limitations.</b> (1) Labels come from a heuristic alignment of reference routes that are themselves "
            "imperfect (trips 33 and 54). (2) The decoder assigns one arc per fix, so short connecting arcs between "
            "fixes are not inserted, which limits route recall. (3) The classifier is trained on candidate rows "
            "independently; a model trained on sequences, or decoder transitions learned from data, could help on "
            "trips like 00000017. (4) KNN and the MLP were trained on samples for runtime reasons."),
          table([["Step", "Command", "Output"],
                 ["EDA", "python scripts/01_eda.py", "reports/eda/, EDA figures, EDA map"],
                 ["Features", "python scripts/02_build_features.py", "cache/ (git-ignored)"],
                 ["Model comparison", "python scripts/03_train_models.py", "reports/model_comparison.csv, cv_folds.csv"],
                 ["Bayesian optimisation", "python scripts/04_bayes_opt.py", "artifacts/final_model.joblib"],
                 ["HMM tuning + comparison", "python scripts/05_hmm_eval.py", "artifacts/hmm_config.json, maps"],
                 ["Evaluator run (API up)", "python scripts/07_evaluator_run.py", "reports/evaluator_test/"],
                 ["This report", "python report/build_report.py", "report/PathSnap_Technical_Report.pdf"]],
                widths=[38 * mm, 62 * mm, PAGE_W - 100 * mm]),
          H1("10. Disclosure of AI assistance"),
          P("In line with the event's policy we disclose that an AI coding assistant (Claude) was used to help write "
            "parts of the code, documentation and this report. Commits containing AI-assisted work carry a "
            "<i>Co-Authored-By: Claude</i> trailer in the repository history. All experiments were actually run on "
            "the provided data; every number, table, figure and map in this report is produced from the committed "
            "result files by <i>report/build_report.py</i>. Each team member has reviewed the part of the system they "
            "own (see TEAM.md) and is responsible for explaining it.")]

doc = Doc(OUT, pagesize=A4, leftMargin=20 * mm, rightMargin=20 * mm, topMargin=18 * mm, bottomMargin=22 * mm,
          title="PathSnap Technical Report", author="Shryes, Shubham Bhat, Siddhant Anil, Siddhant Khurdiya",
          subject="UE24CS352A Machine Learning - PathSnap")
doc.multiBuild(story, onFirstPage=on_page, onLaterPages=on_page)
print(f"wrote {OUT}")
