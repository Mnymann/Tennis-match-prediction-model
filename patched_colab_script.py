#@title USE THIS CODE - PREDICTION + EXCHANGE + EV-FIRST STAKING + NAME ALIGNMENT + 3-MODEL CONSENSUS (SG/G/TA) — WITH SERIES/TOUR/HAND PATCHES + EXCHANGE NEEDED ODDS + DUAL SELECTED-BETS (Main & SG-as-Main)
# ──────────────────────────────────────────────────────────────────────────────
# 0) INSTALLS
# ──────────────────────────────────────────────────────────────────────────────
!pip install -q pandas numpy requests scikit-learn lightgbm catboost
!pip install -q tensorflow==2.16.2 keras==2.16 || true
import os, io, pickle, warnings, math, shutil
from collections import defaultdict, deque
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import requests
import tensorflow as tf
warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# 0.5) MOUNT DRIVE (artifacts live here)
# ──────────────────────────────────────────────────────────────────────────────
from google.colab import drive
drive.mount("/content/drive", force_remount=True)
DRIVE_PROJECT_DIR = "/content/drive/MyDrive/TennisOPENAI"  # ← same folder as training cell
os.makedirs(DRIVE_PROJECT_DIR, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────────
# 1) CONFIG
# ──────────────────────────────────────────────────────────────────────────────
ODDS_CSV_URL    = "https://docs.google.com/spreadsheets/d/e/2PACX-1vSC1gQb41w3qnVCWY3HECkTlMrLQbXhekUXB4IPKvxekwYgyybjuJzMJtNkZ29vgCKSWwEjgMzaL__j/pub?gid=1176846466&single=true&output=csv"
NAMES_CSV_URL   = "https://docs.google.com/spreadsheets/d/e/2PACX-1vTa6K3dnBRvnU00BDCYMNWurBTWzeIMAN-mYP1rbB-T5z3xp5xvvIhiQHwRbYzqGdLRFPV8qRt0ETWM/pub?gid=1773170989&single=true&output=csv"

# Local backup of the odds sheet (optional but recommended)
ODDS_UPLOAD_PATH = "/content/2.Tennis_odds - Upload.csv"
UPLOAD_REPLACE_TOL = 0.05 # if |p_online - p_upload| ≥ 5% on ≥2 sources, prefer upload

# 🔽 Load artifacts from Google Drive (produced by your training cell)
ARTIFACTS_PATH  = f"{DRIVE_PROJECT_DIR}/tennis_core_artifacts.pkl"
DATASET_CSV     = "/content/Sackmann.csv"              # same file used in training (keep local or update if needed)
EXCH_SNAPSHOT   = "/content/exch_snapshot.csv"         # optional Betfair snapshot

# Bankroll & staking
BANKROLL_DKK        = 10_000.0
KELLY_FRACTION        = 0.50
MAX_STAKE_PCT         = 0.05
TOTAL_RISK_CAP_PCT    = 0.25
ROUND_TO              = 10.0

# Leakage & rest guards (match training)
RAW_ELO_CUTOFF      = pd.Timestamp("2017-07-01")
REST_PRIOR_MEAN_DAYS = 7
REST_SHRINK_ALPHA   = 3.0
CLIP_DAYS_ABS_DEFAULT= 21  # fallback if not found in artifacts

# Data-quality gates (auto-drop from predictions + betting plan)
ENFORCE_DATA_QUALITY = True
MIN_MATCHES_TOTAL    = 12
MIN_MATCHES_SURF     = 3

# ================== EXTERNAL SIGNALS (CONSENSUS) ==================
USE_EXTERNAL_SIGNALS  = True
USE_EXTERNAL_IN_PROB  = True    # ✅ blend externals into actual probabilities (tempered logit)

# *** TP REMOVED. We only use SG/G/TA. ***
EXT_WEIGHTS = {
    "steveg":           0.46,
    "gemini":           0.29,
    "tennisabstract":   0.25,
}

# source lists (name -> tag)
EXT_SOURCES_BASE = [("gemini","G"), ("tennisabstract","TA"), ("steveg","SG")]        # for normal plan (main=our model)
EXT_SOURCES_SWAP = [("gemini","G"), ("tennisabstract","TA"), ("op","OP")]            # for SG-as-main plan (externals include OP)

EXT_TEMP            = 1.6
EXT_BLEND_W_MODEL = 0.78
EDGE_MIN = 0.03

# === EV-first consensus parameters ===
VALUE_AGREE_MIN          = 2        # out of 3 now
VALUE_AVG_EDGE_STRONG    = 0.03
CONSENSUS_THR_BONUS      = 0.005
CONSENSUS_KELLY_BOOST    = 1.15
OPP_VALUE_VETO_EDGE      = 0.06
OPP_VALUE_SOFT_EDGE      = 0.03
DISAGREE_KELLY_CUT       = 0.60
STD_EDGE_MAX             = 0.10
STD_CUT_FACTOR           = 0.25
EDGE_INTENSITY_REF       = 0.08
INTENSITY_KELLY_BOOST    = 0.15
INTENSITY_THR_BONUS      = 0.003

# 🔁 Exchange commission (applied to winnings)
COMMISSION_EXCH = 0.065

def _ev_min_for_odds(odds: float) -> float:
    if odds >= 6.0: return 0.06
    if odds >= 3.0: return 0.03
    if odds >= 2.0: return 0.02
    return 0.01

def _thr_for_odds(odds, base):
    thr = base
    if odds >= 5.0:  thr = max(thr, 0.08)
    if odds <= 1.15: thr = max(thr, 0.02)
    return thr

# ──────────────────────────────────────────────────────────────────────────────
# 2) HELPERS — PATCHED
# ──────────────────────────────────────────────────────────────────────────────
def _fetch_csv(url):
    r = requests.get(url, headers={"User-Agent":"Mozilla/5.0"}, timeout=60); r.raise_for_status()
    return pd.read_csv(io.StringIO(r.text))

def _clean_name(s):
    if s is None: return np.nan
    s2 = str(s).strip()
    return np.nan if s2 == "" else s2

def _clean_odds(x):
    if pd.isna(x): return np.nan
    x = str(x).strip().replace(",", ".")
    try:
        v = float(x)
        return v if v > 1.01 and np.isfinite(v) else np.nan
    except Exception:
        return np.nan

# Surface clamp (Hard/Clay/Grass only)
def _norm_surface(s):
    t = str(s).strip().lower()
    if "grass" in t: return "Grass"
    if "clay"  in t: return "Clay"
    # includes carpet/acrylic/unknown → Hard
    return "Hard"

# Hand normalization (R/L only; else→R)
def _norm_hand(x):
    t = str(x).strip().upper()
    return "L" if t == "L" else "R"

# Series/Tour → tier mapping
_SERIES_ALIASES = {
    "g": "GS", "grand slam": "GS",
    "m": "M",  "masters": "M", "masters 1000": "M", "atp masters 1000": "M",
    "a": None, "atp": None,
    "atp500": "500", "500": "500",
    "atp250": "250", "250": "250",
    "c": "250", "challenger": "250",
    "o": "Other", "olympics": "Other",
    "f": "Other", "finals": "Other", "tour finals": "Other", "future finals": "Other",
}
def _series_tier_from_fields(series_val, tour_val):
    s_raw = str(series_val).strip().lower()
    t_raw = str(tour_val).strip().lower()
    if s_raw in _SERIES_ALIASES and _SERIES_ALIASES[s_raw] is not None:
        return _SERIES_ALIASES[s_raw]
    for key, tier in [("grand", "GS"), ("masters", "M"), ("1000","M"),
                      ("500","500"), ("250","250"), ("challenger","250"),
                      ("olympic","Other"), ("final","Other")]:
        if key in s_raw:
            return tier
    if "grand" in t_raw:   return "GS"
    if "master" in t_raw:  return "M"
    if "olympic" in t_raw: return "Other"
    if "final" in t_raw:   return "Other"
    if "challenger" in t_raw: return "250"
    return "Other"

def _clean_prob(x):
    if pd.isna(x): return np.nan
    s = str(x).strip().replace("%","").replace(",",".")
    if s == "": return np.nan
    try:
        v = float(s)
        if v > 1.0: v = v / 100.0
        return float(np.clip(v, 1e-4, 1-1e-4))
    except Exception:
        return np.nan

def read_names_map(url):
    try:
        df = _fetch_csv(url)
    except Exception as e:
        print(f"⚠️ Could not fetch Names sheet, proceeding with identity mapping: {e}")
        return {}
    masters = df.iloc[:,0].astype(str).str.strip()
    aliases = df.iloc[:,6].astype(str).str.strip()
    m = {}
    for a, b in zip(aliases, masters):
        if str(a).strip() and str(b).strip():
            m[a.strip().lower()] = b.strip()
            m[b.strip().lower()] = b.strip()
    return m

def resolve_name(name, name_map):
    t = str(name).strip()
    return name_map.get(t.lower(), t)

def _canon(s): return str(s).strip()
def _match_id(tourney, surface, p1, p2): # order-specific: p1 vs p2 (we’ll also check the swapped id when reconciling)
    return "|".join([
        _canon(tourney).casefold(),
        _norm_surface(surface),
        resolve_name(p1, {}), # resolve later; id here is only for initial join
        resolve_name(p2, {})
    ])

def _read_upload_csv(path):
    if not os.path.exists(path):
        return None, {}
    try:
        up = pd.read_csv(path)
    except Exception as e:
        print(f"⚠️ Could not read upload CSV: {e}")
        return None, {}
    # try to detect column names like the online sheet
    colmap = {
        "p1":"p1", "p2":"p2", "odds1":"odds1", "odds2":"odds2", "surface":"surface","tour":"tour","tourney":"tourney",
        "p1_gemini":"p1_gemini","p2_gemini":"p2_gemini",
        "p1_tennisabstract":"p1_tennisabstract","p2_tennisabstract":"p2_tennisabstract",
        "p1_steveg":"p1_steveg","p2_steveg":"p2_steveg",
    }
    missing = [c for c in ["p1","p2","surface","tourney"] if c not in up.columns]
    if missing:
        print(f"⚠️ Upload CSV missing key cols {missing}; will skip reconcile.")
        return up, {}
    # build lookup by canonical id (both orders)
    lookup = {}
    for _, r in up.iterrows():
        p1, p2 = _clean_name(r["p1"]), _clean_name(r["p2"])
        srf = _norm_surface(r["surface"])
        tny = str(r["tourney"]).strip()
        # raw probs (may be prob or %) → _clean_prob
        def gget(a,b): return _clean_prob(r.get(f"{a}_{b}", np.nan))
        g1, g2 = gget("p1","gemini"), gget("p2","gemini")
        t1, t2 = gget("p1","tennisabstract"), gget("p2","tennisabstract")
        s1, s2 = gget("p1","steveg"), gget("p2","steveg")
        if p1 and p2:
            k = _match_id(tny, srf, p1, p2).casefold()
            kr = _match_id(tny, srf, p2, p1).casefold()
            lookup[k] = {"p1_g":g1,"p2_g":g2,"p1_ta":t1,"p2_ta":t2,"p1_sg":s1,"p2_sg":s2}
            lookup[kr] = {"p1_g":g2,"p2_g":g1,"p1_ta":t2,"p2_ta":t1,"p1_sg":s2,"p2_sg":s1}
    return up, lookup

def _reconcile_externals_with_upload(odds_df, upload_lookup, name_map):
    """If upload has the same match and externals differ materially, overwrite with upload."""
    if not upload_lookup:
        return odds_df, 0
    fixed = 0
    out = odds_df.copy()
    # build canonical match_id AFTER name resolution
    ids = []
    for _, r in out.iterrows():
        p1 = resolve_name(r["p1_name"], name_map)
        p2 = resolve_name(r["p2_name"], name_map)
        ids.append(_match_id(r["tourney_name"], r["surface"], p1, p2).casefold())
    out["match_id"] = ids
    for i, r in out.iterrows():
        k = r["match_id"]
        cand = upload_lookup.get(k)
        if not cand: continue
        # online probs
        online = {
            "p1_g": r.get("ext_gemini_p1", np.nan), "p2_g": r.get("ext_gemini_p2", np.nan),
            "p1_ta": r.get("ext_tennisabstract_p1", np.nan), "p2_ta": r.get("ext_tennisabstract_p2", np.nan),
            "p1_sg": r.get("ext_steveg_p1", np.nan), "p2_sg": r.get("ext_steveg_p2", np.nan),
        }
        # count sources where both present and differ ≥ tol
        diffs = 0; total = 0
        for key in online.keys():
            o = online[key]; u = cand.get(key, np.nan)
            if pd.notna(o) and pd.notna(u):
                total += 1
                if abs(float(o) - float(u)) >= UPLOAD_REPLACE_TOL:
                    diffs += 1
        if total >= 2 and diffs >= 2:
            # replace externals with upload values
            out.at[i, "ext_gemini_p1"] = cand["p1_g"]
            out.at[i, "ext_gemini_p2"] = cand["p2_g"]
            out.at[i, "ext_tennisabstract_p1"]= cand["p1_ta"]
            out.at[i, "ext_tennisabstract_p2"]= cand["p2_ta"]
            out.at[i, "ext_steveg_p1"] = cand["p1_sg"]
            out.at[i, "ext_steveg_p2"] = cand["p2_sg"]
            out.at[i, "externals_source"] = "UPLOAD_CSV"
            fixed += 1
    return out.drop(columns=["match_id"]), fixed

def _clip01(p): return np.clip(p, 1e-4, 1-1e-4)

# ──────────────────────────────────────────────────────────────────────────────
# 3) ARTIFACTS + NAMES
# ──────────────────────────────────────────────────────────────────────────────
def load_artifacts(path=ARTIFACTS_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Artifacts not found at: {path}\nMake sure the training cell saved to Google Drive.")
    with open(path, "rb") as f:
        art = pickle.load(f)
    needed = ["stack_model","dl_savedmodel_dir","kept_feature_names","cont_seq_names","calibration","roll_n","calendar"]
    for k in needed:
        assert k in art, f"Artifact missing '{k}'. Please retrain with the latest training cell."
    return art

# ──────────────────────────────────────────────────────────────────────────────
# 4) STATES REBUILD (matches training) — Elo + recency + tiers + sequences
# (unchanged from your previous working version, trimmed for brevity)
# ──────────────────────────────────────────────────────────────────────────────
ELO_START = 1500.0
ELO_K     = 24.0
ELO_SETS_K= 28.0
def elo_expected(ra, rb): return 1.0 / (1.0 + 10.0 ** ((rb - ra) / 400.0))
def elo_update(ra, rb, score_a, k=ELO_K):
    ea = elo_expected(ra, rb); ra2 = ra + k*(score_a - ea); rb2 = rb + k*((1.0-score_a) - (1.0-ea))
    return ra2, rb2
def elo_update_sets(ra, rb, sets_a, sets_b, k=ELO_SETS_K):
    tot = max(1.0, float(sets_a)+float(sets_b)); score_a = float(sets_a)/tot
    return elo_update(ra, rb, score_a, k=k)

def build_states_and_h2h(dataset_csv, warmup_start_year=2015):
    df = pd.read_csv(dataset_csv)
    # Dates/Year
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    if "Year" not in df.columns or df["Year"].isna().all():
        df["Year"] = df["Date"].dt.year
    df = df[df["Year"] >= warmup_start_year].copy()
    # Surface clamp
    if "Surface" in df.columns:
        df["Surface"] = df["Surface"].map(_norm_surface)
    # Tour: DO NOT filter to ATP only (per requirement)
    if "Tour" in df.columns:
        df["Tour"] = df["Tour"].astype(str).str.strip()
    # Normalize hands
    for col_old, col_new in [("winner_hand", "winner_hand_norm"),
                             ("loser_hand",  "loser_hand_norm"),
                             ("WinnerHand",  "winner_hand_norm"),
                             ("LoserHand",   "loser_hand_norm")]:
        if col_old in df.columns:
            df[col_new] = df[col_old].map(_norm_hand)
    # Name resolve
    name_map = read_names_map(NAMES_CSV_URL)
    WCOL = next((c for c in ["Winner_sackmann_names","Winner_name_h","Winner","winner_name"] if c in df.columns), None)
    LCOL = next((c for c in ["Loser_sackmann_names","Loser_name_h","Loser","loser_name"] if c in df.columns), None)
    if not WCOL or not LCOL: raise ValueError("Could not find Winner/Loser columns in dataset.")
    df["W"] = df[WCOL].astype(str).map(lambda x: resolve_name(x, name_map))
    df["L"] = df[LCOL].astype(str).map(lambda x: resolve_name(x, name_map))
    roll_n = 30
    def new_state():
        return {
            "elo_all":ELO_START,
            "elo_surf":{"Hard":ELO_START,"Clay":ELO_START,"Grass":ELO_START,"Other":ELO_START},
            "elo_sets_all":ELO_START,
            "elo_sets_surf":{"Hard":ELO_START,"Clay":ELO_START,"Grass":ELO_START,"Other":ELO_START},
            "wins_all":deque(maxlen=roll_n),
            "wins_surf":{"Hard":deque(maxlen=roll_n),"Clay":deque(maxlen=roll_n),"Grass":deque(maxlen=roll_n),"Other":deque(maxlen=roll_n)},
            "sets_w_all":deque(maxlen=roll_n),
            "sets_l_all":deque(maxlen=roll_n),
            "sps_all":deque(maxlen=roll_n),
            "recent_hist":deque(maxlen=40),
            "recent_dates":deque(maxlen=roll_n),
            "recent_surfs":deque(maxlen=roll_n),
            "recent_locs": deque(maxlen=roll_n),
            "tier_elo":{"GS":ELO_START,"M":ELO_START,"500":ELO_START,"250":ELO_START,"Other":ELO_START},
            "quality_weights":deque(maxlen=18),
            "quality_scores":deque(maxlen=18),
            "seq_hist":deque(maxlen=200)
        }
    players = defaultdict(new_state)
    h2h = defaultdict(lambda: defaultdict(int))
    def replay_recent_elo(history, ref_date, max_items=None, days_limit=None):
        r = ELO_START; cnt = 0
        for (d, opp_r, outc) in history:
            if d >= ref_date: continue
            if days_limit is not None and (ref_date - d).days > days_limit: continue
            r, _ = elo_update(r, opp_r if not np.isnan(opp_r) else ELO_START, outc, k=ELO_K)
            cnt += 1
            if max_items is not None and cnt >= max_items: break
        return r
    def winrate(dq): return (sum(dq)/len(dq)) if len(dq) else 0.0
    def sets_wr(wdq, ldq):
        w, l = sum(wdq), sum(ldq)
        return (w/(w+l)) if (w+l)>0 else 0.0
    cont_seq_names = [
        "elo_all","elo_surf","elo_sets_all","elo_sets_surf",
        "wr_all","wr_surf","matches_all","matches_surf",
        "sps_all","sets_wr_all","elo_recent10","elo_90d",
        "quality_wmean","quality_trend"
    ]
    def prematch_vec(P, srf, now):
        wr_all = winrate(P["wins_all"]); wr_s = winrate(P["wins_surf"][srf])
        matches_all = len(P["wins_all"]); matches_s = len(P["wins_surf"][srf])
        sps_all = (sum(P["sps_all"])/len(P["sps_all"])) if len(P["sps_all"]) else 0.0
        setswr_all = sets_wr(P["sets_w_all"], P["sets_l_all"])
        elo_r10 = replay_recent_elo(P["recent_hist"], now, max_items=12)
        elo_90d = replay_recent_elo(P["recent_hist"], now, days_limit=120)
        if len(P["quality_scores"]) and sum(P["quality_weights"]):
            wmean = float(np.dot(list(P["quality_weights"]), list(P["quality_scores"])) / sum(P["quality_weights"]))
        else:
            wmean = 0.0
        trend = 0.0
        if len(P["quality_scores"]) >= 2:
            trend = list(P["quality_scores"])[-1] - wmean
        return np.array([
            P["elo_all"],
            P["elo_surf"][srf],
            P["elo_sets_all"],
            P["elo_sets_surf"][srf],
            wr_all, wr_s, matches_all, matches_s,
            sps_all, setswr_all,
            elo_r10, elo_90d,
            wmean, trend
        ], dtype=np.float32)
    df = df.sort_values(["Date","Surface"]).reset_index(drop=True)
    for _, r in df.iterrows():
        date = pd.to_datetime(r["Date"])
        if pd.isna(date): continue
        srf = _norm_surface(r["Surface"])
        tier = _series_tier_from_fields(r.get("Series",""), r.get("Tour",""))
        w = r["W"]; l = r["L"]
        if not w or not l: continue
        Pw = players[w]; Pl = players[l]
        Pw["seq_hist"].append(prematch_vec(Pw, srf, date))
        Pl["seq_hist"].append(prematch_vec(Pl, srf, date))
        Pw["elo_all"], Pl["elo_all"] = elo_update(Pw["elo_all"], Pl["elo_all"], 1.0, k=ELO_K)
        Pw["elo_surf"][srf], Pl["elo_surf"][srf] = elo_update(Pw["elo_surf"][srf], Pl["elo_surf"][srf], 1.0, k=ELO_K)
        Pw["wins_all"].append(1); Pl["wins_all"].append(0)
        Pw["wins_surf"][srf].append(1); Pl["wins_surf"][srf].append(0)
        if "WinSets" in df.columns and "LoserSets" in df.columns:
            try:
                ws = float(r.get("WinSets", np.nan)); ls = float(r.get("LoserSets", np.nan))
            except Exception:
                ws, ls = np.nan, np.nan
            if pd.notna(ws) and pd.notna(ls):
                Pw["elo_sets_all"], Pl["elo_sets_all"] = elo_update_sets(Pw["elo_sets_all"], Pl["elo_sets_all"], ws, ls, k=ELO_SETS_K)
                Pw["elo_sets_surf"][srf], Pl["elo_sets_surf"][srf] = elo_update_sets(Pw["elo_sets_surf"][srf], Pl["elo_sets_surf"][srf], ws, ls, k=ELO_SETS_K)
                Pw["sets_w_all"].append(ws); Pw["sets_l_all"].append(ls)
                Pl["sets_w_all"].append(ls); Pl["sets_l_all"].append(ws)
        if "W_SetPts_per_set" in df.columns and "L_SetPts_per_set" in df.columns:
            try:
                w_sps = float(r.get("W_SetPts_per_set", np.nan))
                l_sps = float(r.get("L_SetPts_per_set", np.nan))
            except Exception:
                w_sps, l_sps = np.nan, np.nan
            if pd.notna(w_sps): Pw["sps_all"].append(w_sps)
            if pd.notna(l_sps): Pl["sps_all"].append(l_sps)
        Pw["recent_hist"].append((date, Pl["elo_all"], 1.0))
        Pl["recent_hist"].append((date, Pw["elo_all"], 0.0))
        def add_quality(P, opp_elo, won):
            if np.isnan(opp_elo): opp_elo = 1500.0
            weight = max(0.5, (opp_elo - 1500.0)/300.0 + 1.0)
            P["quality_weights"].append(weight)
            P["quality_scores"].append(1.0 if won else 0.0)
        add_quality(Pw, Pl["elo_all"], True)
        add_quality(Pl, Pw["elo_all"], False)
        Pw["tier_elo"][tier], _ = elo_update(Pw["tier_elo"][tier], Pl["tier_elo"][tier], 1.0, k=ELO_K)
        Pl["tier_elo"][tier], _ = elo_update(Pl["tier_elo"][tier], Pw["tier_elo"][tier], 0.0, k=ELO_K)
        Pw["recent_dates"].append(date); Pl["recent_dates"].append(date)
        Pw["recent_surfs"].append(srf);  Pl["recent_surfs"].append(srf)
        h2h[w][l] += 1
    return players, h2h, cont_seq_names, roll_n, name_map

def seq_from_player(P, srf, roll_n, cont_seq_names):
    seq = list(P["seq_hist"])
    if not seq:
        return np.zeros((roll_n, len(cont_seq_names)), dtype=np.float32)
    arr = np.stack(seq[-roll_n:], axis=0)
    if arr.shape[0] < roll_n:
        pad = np.zeros((roll_n - arr.shape[0], arr.shape[1]), dtype=np.float32)
        arr = np.vstack([pad, arr])
    return arr.astype(np.float32)

def last_surface_gap_days(P, surface, now, clip=21):
    last_on = None
    for d, s in zip(reversed(P["recent_dates"]), reversed(P["recent_surfs"])):  # newest at end
        if s == surface and d < now:
            last_on = d; break
    if last_on is None: return clip
    return min(clip, max(-clip, (now - last_on).days))

def recent_matches_7d(P, now):
    return sum(1 for d in P["recent_dates"] if (now - d).days <= 7)

def surf_share3(P, srf):
    seq = list(P["recent_surfs"])[:]
    seq = seq[-3:]
    return (seq.count(srf) / max(1,len(seq))) if seq else 0.0

# ──────────────────────────────────────────────────────────────────────────────
# 5) CALIBRATION / SAVEDMODEL CALL
# ──────────────────────────────────────────────────────────────────────────────
def apply_calibration(method, params, p_raw, surface=None):
    p_raw = _clip01(np.asarray(p_raw))
    lam = float(params.get("lambda", 1.0)) if params else 1.0
    if method == "isotonic_global" and params and "iso_global" in params:
        ir = params["iso_global"]; p_c = _clip01(ir.transform(p_raw))
        return _clip01(lam*p_c + (1-lam)*p_raw)
    if method == "isotonic_per_surface" and params and "iso_by_surface" in params and surface is not None:
        p_c = p_raw.copy()
        for srf, ir in params["iso_by_surface"].items():
            m = (surface == srf) if isinstance(surface, np.ndarray) else (surface == srf)
            if np.any(m):
                p_c[m] = _clip01(ir.transform(p_raw[m]))
        return _clip01(lam*p_c + (1-lam)*p_raw)
    if method == "platt" and params and "platt" in params:
        lr = params["platt"]; p_c = _clip01(lr.predict_proba(p_raw.reshape(-1,1))[:,1])
        return _clip01(lam*p_c + (1-lam)*p_raw)
    if method == "temperature" and params and "temperature" in params:
        ts = params["temperature"]
        logits = np.log(p_raw) - np.log(1-p_raw)
        p_c = _clip01(1/(1+np.exp(-logits/float(getattr(ts,'T_',1.0)))) )
        return _clip01(lam*p_c + (1-lam)*p_raw)
    return p_raw

def call_savedmodel_flex(serve_fn, p1_seq, p2_seq):
    p1 = tf.convert_to_tensor(p1_seq)
    p2 = tf.convert_to_tensor(p2_seq)
    try:
        out = serve_fn(p1_seq=p1, p2_seq=p2)
    except TypeError:
        out = serve_fn(p1, p2)
    if hasattr(out, "numpy"):
        return out.numpy().ravel()
    if isinstance(out, dict):
        for k in ["match_proba", "prob", "output_0", "outputs", "dense", "Identity"]:
            if k in out:
                v = out[k]
                return (v.numpy().ravel() if hasattr(v, "numpy") else np.array(v).ravel())
        v = next(iter(out.values()))
        return (v.numpy().ravel() if hasattr(v, "numpy") else np.array(v).ravel())
    return np.array(out).ravel()

# ──────────────────────────────────────────────────────────────────────────────
# 6) EXCHANGE + SOURCE PICK
# ──────────────────────────────────────────────────────────────────────────────
def load_exchange_snapshot_csv(path=EXCH_SNAPSHOT):
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
    for c in ["sheet_tourney_name","master_home","master_away","exch_home_net","exch_away_net"]:
        if c not in df.columns: df[c] = np.nan
    return df

def add_exchange_to_preds(pred_df, exch_df):
    out = pred_df.copy()
    for c in ["exch_p1","exch_p2","ev_exch_p1","ev_exch_p2",
              "imp_exch_p1","imp_exch_p2","edge_exch_p1","edge_exch_p2",
              "best_source_p1","best_source_p2","best_odds_p1","best_odds_p2",
              "best_ev_p1","best_ev_p2","exch_pair_ok"]:
        if c not in out.columns:
            out[c] = np.nan
    out["best_source_p1"] = out["best_source_p1"].fillna("BOOK")
    out["best_source_p2"] = out["best_source_p2"].fillna("BOOK")
    out["exch_pair_ok"]   = out["exch_pair_ok"].fillna(False)
    if exch_df.empty:
        out["best_odds_p1"] = out["p1_odds"]; out["best_odds_p2"] = out["p2_odds"]
        out["best_ev_p1"]   = out["ev_book_p1"]; out["best_ev_p2"] = out["ev_book_p2"]
        return out
    def _k(tn, a, b): return (str(tn).strip().casefold(), str(a).strip(), str(b).strip())
    key_to_exch = {}
    for _, r in exch_df.iterrows():
        tn = str(r["sheet_tourney_name"])
        mh = r["master_home"]; ma = r["master_away"]
        p1 = float(r["exch_home_net"]) if pd.notna(r["exch_home_net"]) else np.nan
        p2 = float(r["exch_away_net"]) if pd.notna(r["exch_away_net"]) else np.nan
        key_to_exch[_k(tn, mh, ma)] = (p1, p2)
        key_to_exch[_k(tn, ma, mh)] = (p2, p1)
    for i, r in out.iterrows():
        k = _k(str(r["tourney_name"]), r["p1_name"], r["p2_name"])
        if k in key_to_exch:
            p1_net, p2_net = key_to_exch[k]
            out.at[i,"exch_p1"] = p1_net
            out.at[i,"exch_p2"] = p2_net
            out.at[i,"exch_pair_ok"] = bool(pd.notna(p1_net) and pd.notna(p2_net))
    out["ev_exch_p1"] = out["prob_p1"] * out["exch_p1"] - 1
    out["ev_exch_p2"] = out["prob_p2"] * out["exch_p2"] - 1
    with np.errstate(divide="ignore", invalid="ignore"):
        imp1_raw = 1.0 / out["exch_p1"]; imp2_raw = 1.0 / out["exch_p2"]
        s = imp1_raw + imp2_raw
        out["imp_exch_p1"] = imp1_raw / s
        out["imp_exch_p2"] = imp2_raw / s
    out["edge_exch_p1"] = out["prob_p1"] - out["imp_exch_p1"]
    out["edge_exch_p2"] = out["prob_p2"] - out["imp_exch_p2"]
    def pick(book_ev, book_odds, exch_ev, exch_odds):
        ev_b = book_ev if pd.notna(book_ev) else -np.inf
        ev_x = exch_ev if pd.notna(exch_ev) else -np.inf
        if pd.notna(exch_odds) and ev_x >= ev_b:
            return "EXCH", float(exch_odds), float(ev_x)
        else:
            return "BOOK", float(book_odds), float(ev_b)
    picks = out.apply(lambda r: pd.Series({
        "best_source_p1": pick(r["ev_book_p1"], r["p1_odds"], r["ev_exch_p1"], r["exch_p1"])[0],
        "best_odds_p1":   pick(r["ev_book_p1"], r["p1_odds"], r["ev_exch_p1"], r["exch_p1"])[1],
        "best_ev_p1":     pick(r["ev_book_p1"], r["p1_odds"], r["ev_exch_p1"], r["exch_p1"])[2],
        "best_source_p2": pick(r["ev_book_p2"], r["p2_odds"], r["ev_exch_p2"], r["exch_p2"])[0],
        "best_odds_p2":   pick(r["ev_book_p2"], r["p2_odds"], r["ev_exch_p2"], r["exch_p2"])[1],
        "best_ev_p2":     pick(r["ev_book_p2"], r["p2_odds"], r["ev_exch_p2"], r["exch_p2"])[2],
    }), axis=1)
    for c in picks.columns: out[c] = picks[c]
    return out

# ──────────────────────────────────────────────────────────────────────────────
# 7) STAKING / EXTERNALS — generalized to support different source sets
# ──────────────────────────────────────────────────────────────────────────────
def kelly_fraction(p, odds):
    b = odds - 1.0
    if b <= 0: return 0.0
    f = (p*odds - 1.0) / b
    return max(0.0, f)

def _chosen_imp_from_row(r, side):
    return r.get(f"imp_exch_{side}", np.nan) if r.get(f"best_source_{side}", "BOOK") == "EXCH" else r.get(f"imp_book_{side}", np.nan)

def _needed_exchange_odds(book_odds, commission=COMMISSION_EXCH):
    if pd.isna(book_odds): return np.nan
    return 1.0 + (float(book_odds) - 1.0) / max(1e-9, (1.0 - commission))

def _ext_passes_strict(r, side, p_ext, chosen_odds):
    if pd.isna(p_ext) or pd.isna(chosen_odds): return False
    imp = _chosen_imp_from_row(r, side)
    if pd.isna(imp): return False
    thr    = _thr_for_odds(float(chosen_odds), EDGE_MIN)
    ev_min = _ev_min_for_odds(float(chosen_odds))
    edge = float(p_ext) - float(imp)
    ev   = float(p_ext) * float(chosen_odds) - 1.0
    return (edge >= thr) and (ev >= ev_min)

def _ext_value_stats_generic(r, side, ext_sources):
    imp  = _chosen_imp_from_row(r, side)
    odds = r.get(f"best_odds_{side}", np.nan)
    if pd.isna(imp) or pd.isna(odds):
        return 0.0, 0.0, 0, 0.0
    edges, wts = [], []
    agree_cnt = 0
    for src, _code in ext_sources:
        vname = f"ext_{src}_{side}" if src != "op" else f"{src}_{side}"
        p = r.get(vname, np.nan)
        if pd.notna(p):
            p = float(p)
            edges.append(p - float(imp))
            wts.append(EXT_WEIGHTS.get(src, 0.0)) # OP not in weights → 0
            if _ext_passes_strict(r, side, p, odds):
                agree_cnt += 1
    if not edges:
        return 0.0, 0.0, 0, 0.0
    wts = np.asarray(wts if sum(wts)>0 else [1.0]*len(edges), float)
    wts = wts / max(1e-9, wts.sum())
    edges = np.asarray(edges, float)
    mean_edge = float((wts*edges).sum())
    var = float((wts*((edges - mean_edge)**2)).sum())
    std_edge = float(np.sqrt(max(0.0, var)))
    # opposite side
    opp_side = "p2" if side=="p1" else "p1"
    opp_imp  = _chosen_imp_from_row(r, opp_side)
    opp_edges, opp_wts = [], []
    for src, _code in ext_sources:
        vname = f"ext_{src}_{opp_side}" if src != "op" else f"{src}_{opp_side}"
        p = r.get(vname, np.nan)
        if pd.notna(p) and pd.notna(opp_imp):
            opp_edges.append(float(p) - float(opp_imp))
            opp_wts.append(EXT_WEIGHTS.get(src, 0.0))
    if opp_edges:
        opp_wts = np.array(opp_wts if sum(opp_wts)>0 else [1.0]*len(opp_edges), float)
        opp_wts = opp_wts / max(1e-9, opp_wts.sum())
        opp_mean_edge = float((opp_wts*np.array(opp_edges,float)).sum())
    else:
        opp_mean_edge = 0.0
    return mean_edge, std_edge, int(agree_cnt), opp_mean_edge

def _ext_tagline_with_odds(r, player_name, ext_sources, chosen_side=None):
    # Identify side + chosen odds
    if r.get("p1_name") == player_name:
        side = "p1"
        odds = r.get("best_odds_p1", np.nan)
    elif r.get("p2_name") == player_name:
        side = "p2"
        odds = r.get("best_odds_p2", np.nan)
    else:
        return f"value_agree=0/3 [" + ", ".join([f"{code}:–" for _,code in ext_sources]) + "]"
    tags = []
    agree = 0
    for src, code in ext_sources:
        vname = f"ext_{src}_{side}" if src != "op" else f"{src}_{side}"
        p = r.get(vname, np.nan)
        if pd.notna(p):
            dec_odds = (1.0/float(p)) if p>0 else np.nan
            ok = _ext_passes_strict(r, side, float(p), odds)
            if pd.notna(dec_odds):
                tags.append(f"{code}: {dec_odds:.2f}{'✅' if ok else '❌'}")
            else:
                tags.append(f"{code}: –")
            if ok: agree += 1
        else:
            tags.append(f"{code}: –")
    return f"value_agree={agree}/3 [" + ", ".join(tags) + "]", agree

def build_betting_plan_generic(pred_df, bankroll, prob_cols=("prob_p1","prob_p2"), ext_sources=EXT_SOURCES_BASE):
    rows=[]
    for _, r in pred_df.iterrows():
        p1 = float(r.get(prob_cols[0], np.nan))
        p2 = float(r.get(prob_cols[1], np.nan))
        if pd.isna(p1) or pd.isna(p2):
            continue
        o1, o2 = float(r["best_odds_p1"]), float(r["best_odds_p2"])
        # recompute book EV with the given probs (prob_cols)
        ev_book_p1 = p1 * float(r["p1_odds"]) - 1
        ev_book_p2 = p2 * float(r["p2_odds"]) - 1
        ev1 = p1 * o1 - 1
        ev2 = p2 * o2 - 1
        imp1 = _chosen_imp_from_row(r, "p1"); imp2 = _chosen_imp_from_row(r, "p2")
        thr1 = _thr_for_odds(o1, EDGE_MIN); thr2 = _thr_for_odds(o2, EDGE_MIN)
        k1 = kelly_fraction(p1, o1) * KELLY_FRACTION
        k2 = kelly_fraction(p2, o2) * KELLY_FRACTION
        # externals (generic lists)
        m1, s1, a1, opp1 = _ext_value_stats_generic(r, "p1", ext_sources)
        m2, s2, a2, opp2 = _ext_value_stats_generic(r, "p2", ext_sources)
        def _adj(thr, k, mean_edge, std_edge, agree_cnt, opp_mean_edge):
            thr_adj, k_adj, veto = thr, k, False
            if agree_cnt >= VALUE_AGREE_MIN and mean_edge >= VALUE_AVG_EDGE_STRONG:
                thr_adj = max(0.0, thr_adj - CONSENSUS_THR_BONUS)
                k_adj   = k_adj * CONSENSUS_KELLY_BOOST
            intensity = max(0.0, min(1.0, mean_edge / EDGE_INTENSITY_REF))
            k_adj   *= (1.0 + INTENSITY_KELLY_BOOST * intensity)
            thr_adj  = max(0.0, thr_adj - INTENSITY_THR_BONUS * intensity)
            disp = max(0.0, min(1.0, std_edge / STD_EDGE_MAX))
            k_adj *= (1.0 - STD_CUT_FACTOR * disp)
            if opp_mean_edge >= OPP_VALUE_VETO_EDGE:
                veto = True
            elif opp_mean_edge >= OPP_VALUE_SOFT_EDGE:
                k_adj *= DISAGREE_KELLY_CUT
            return thr_adj, k_adj, veto
        thr1, k1, veto1 = _adj(thr1, k1, m1, s1, a1, opp1)
        thr2, k2, veto2 = _adj(thr2, k2, m2, s2, a2, opp2)
        edge1 = p1 - (imp1 if pd.notna(imp1) else 0.0)
        edge2 = p2 - (imp2 if pd.notna(imp2) else 0.0)
        ev_min1 = _ev_min_for_odds(o1); ev_min2 = _ev_min_for_odds(o2)
        p1_ok = (ev1 >= ev_min1) and (edge1 >= thr1) and (not veto1)
        p2_ok = (ev2 >= ev_min2) and (edge2 >= thr2) and (not veto2)
        if not (p1_ok or p2_ok):
            continue
        # Tag lines with decimal-odds display next to each external (and ✅/❌)
        tag1, agree_tag1 = _ext_tagline_with_odds(r, r["p1_name"], ext_sources)
        tag2, agree_tag2 = _ext_tagline_with_odds(r, r["p2_name"], ext_sources)
        choices=[]
        if p1_ok:
            choices.append(("p1", k1, ev1, edge1, thr1, r["p1_name"], o1, p1,
                            r["best_source_p1"], tag1, agree_tag1))
        if p2_ok:
            choices.append(("p2", k2, ev2, edge2, thr2, r["p2_name"], o2, p2,
                            r["best_source_p2"], tag2, agree_tag2))
        # pick highest k*EV
        choices.sort(key=lambda x: (x[1]*x[2]), reverse=True)
        side, k_frac, ev, edge, thr_eff, name, odds_chosen, prob, source, ext_tag, agree_cnt = choices[0]
        book_odds_side = float(r["p1_odds"] if side=="p1" else r["p2_odds"])
        exch_needed_side = _needed_exchange_odds(book_odds_side, COMMISSION_EXCH)
        exch_actual_side = float(r.get("exch_p1" if side=="p1" else "exch_p2", np.nan)) if pd.notna(r.get("exch_p1" if side=="p1" else "exch_p2", np.nan)) else np.nan
        stake_frac = min(k_frac, MAX_STAKE_PCT)
        stake_amt  = bankroll * stake_frac
        if ROUND_TO and stake_amt>0: stake_amt = np.floor(stake_amt/ROUND_TO)*ROUND_TO
        rows.append({
            "tourney_name": r["tourney_name"], "surface": r["surface"], "tour": r["tour"],
            "p1_name": r["p1_name"], "p2_name": r["p2_name"],
            "player": name, "opponent": r["p2_name"] if side=="p1" else r["p1_name"],
            "chosen_side": side, "source": source, "odds": float(odds_chosen), "model_prob": float(prob),
            "implied_prob": float(_chosen_imp_from_row(r, side)),
            "edge_devig": float(edge), "edge_threshold": float(thr_eff),
            "kelly_fraction_applied": float(k_frac),
            "stake_frac_of_bankroll": float(stake_frac),
            "stake_dkk": float(stake_amt),
            "ev_decimal": float(ev),
            "expected_profit_dkk": float(stake_amt * ev),
            "market": "Match odds",
            "ext_support": ext_tag,
            "ext_agree_value": int(agree_cnt),
            "book_odds_side": book_odds_side,
            "exch_needed_side": float(exch_needed_side),
            "exch_actual_side": exch_actual_side,
            "blend_prob": float(prob),
            "externals_source": r.get("externals_source")
        })
    plan = pd.DataFrame(rows)
    if plan.empty: return plan, bankroll, 0.0
    total = plan["stake_dkk"].sum()
    cap_amount = bankroll * TOTAL_RISK_CAP_PCT
    if total > cap_amount and total>0:
        scale = cap_amount/total
        plan["stake_dkk"] *= scale
        plan["stake_frac_of_bankroll"] *= scale
        plan["expected_profit_dkk"] *= scale
    plan = plan.sort_values(["expected_profit_dkk","edge_devig","ev_decimal"], ascending=False).reset_index(drop=True)
    return plan, bankroll, plan["expected_profit_dkk"].sum()

# ──────────────────────────────────────────────────────────────────────────────
# 8) MAIN RUN — prints dual Selected Bets (Main, then SG-as-Main)
# ──────────────────────────────────────────────────────────────────────────────
def run():
    print("🔧 Loading artifacts…")
    print(f"   → {ARTIFACTS_PATH}")
    art = load_artifacts(ARTIFACTS_PATH)
    stack = art["stack_model"]
    saved_dir = art["dl_savedmodel_dir"]  # Drive path
    kept_feats = art["kept_feature_names"]
    cont_names = art["cont_seq_names"]
    roll_n = int(art.get("roll_n", 30))
    calib = art.get("calibration", {"method":"none","params":{}})
    method = calib.get("method","none"); params = calib.get("params",{})
    CLIP_DAYS_ABS = art.get("toggles", {}).get("CLIP_DAYS_ABS", CLIP_DAYS_ABS_DEFAULT)
    print("🧠 Loading DL SavedModel…")
    print(f"   → {saved_dir}")
    dl = tf.saved_model.load(saved_dir)
    serve = dl.signatures["serve"]
    print("📥 Loading odds…")
    df = _fetch_csv(ODDS_CSV_URL).dropna(how="all")
    # Try loading local backup to reconcile externals later
    upload_df, upload_lookup = _read_upload_csv(ODDS_UPLOAD_PATH)
    required = ["p1","p2","odds1","odds2","surface","tour","tourney"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Odds CSV missing columns: {missing}")
    date_col = None
    for cand in ["date","match_date","start_time","Start_Time","Start time"]:
        if cand in df.columns: date_col = cand; break
    odds = pd.DataFrame({
        "p1_name": df["p1"].map(_clean_name),
        "p2_name": df["p2"].map(_clean_name),
        "p1_odds": df["odds1"].map(_clean_odds),
        "p2_odds": df["odds2"].map(_clean_odds),
        "surface": df["surface"].map(_norm_surface),
        "tour":    df["tour"].astype(str).str.strip().str.upper(),
        "tourney_name": df["tourney"].astype(str).str.strip().replace("", "Unknown Open"),
    })
    if date_col:
        ev = pd.to_datetime(df[date_col], errors="coerce", utc=True).fillna(pd.Timestamp.now(tz="UTC"))
    else:
        ev = pd.Series(pd.Timestamp.now(tz="UTC"), index=df.index)
    odds["event_date"] = ev.dt.tz_localize(None)
    odds = odds.dropna(subset=["p1_name","p2_name","p1_odds","p2_odds"])
    odds = odds[odds["p1_name"]!=odds["p2_name"]].drop_duplicates().reset_index(drop=True)
    # ---------- External probabilities ----------
    def _get_pair_probs(df_src, base, name):
        c1, c2 = f"p1_{name}", f"p2_{name}"
        if c1 in df_src.columns and c2 in df_src.columns:
            base[f"ext_{name}_p1"] = df_src[c1].map(_clean_prob)
            base[f"ext_{name}_p2"] = df_src[c2].map(_clean_prob)
        else:
            base[f"ext_{name}_p1"] = np.nan
            base[f"ext_{name}_p2"] = np.nan
    if USE_EXTERNAL_SIGNALS:
        for src in ["gemini","tennisabstract","steveg"]:  # TP removed
            _get_pair_probs(df, odds, src)
        # weighted aggregate of available externals (SG/G/TA)
        def _wmean(row, suffix):
            vals, wts = [], []
            for src, w in EXT_WEIGHTS.items():
                v = row.get(f"ext_{src}_{suffix}", np.nan)
                if pd.notna(v):
                    vals.append(v); wts.append(w)
            if not vals: return np.nan
            wts = np.asarray(wts, float); wts = wts / wts.sum()
            return float(np.dot(np.asarray(vals, float), wts))
        odds["ext_p1_agg"] = odds.apply(lambda r: _wmean(r, "p1"), axis=1)
        odds["ext_p2_agg"] = odds.apply(lambda r: _wmean(r, "p2"), axis=1)
        def _present_count(row, suffix):
            return sum(1 for src in ["steveg","gemini","tennisabstract"]
                       if pd.notna(row.get(f"ext_{src}_{suffix}", np.nan)))
        odds["ext_present_p1"] = odds.apply(lambda r: _present_count(r, "p1"), axis=1)
        odds["ext_present_p2"] = odds.apply(lambda r: _present_count(r, "p2"), axis=1)
        odds["ext_consensus_ready"] = True
    else:
        for c in ["ext_gemini_p1","ext_gemini_p2","ext_tennisabstract_p1","ext_tennisabstract_p2","ext_steveg_p1","ext_steveg_p2","ext_p1_agg","ext_p2_agg","ext_present_p1","ext_present_p2","ext_consensus_ready"]:
            if c not in odds.columns: odds[c] = np.nan
    print("📚 Replaying dataset to build states…")
    players, h2h, cont_seq_train, roll_train, name_map = build_states_and_h2h(DATASET_CSV, warmup_start_year=2015)
    # Resolve betting names to canonical (same as training)
    odds["p1_name"] = odds["p1_name"].map(lambda x: resolve_name(x, name_map))
    odds["p2_name"] = odds["p2_name"].map(lambda x: resolve_name(x, name_map))
    print("🧾 Names resolved.")

    # 🔁 Reconcile externals with local upload if they materially disagree
    odds, n_fixed = _reconcile_externals_with_upload(odds, upload_lookup, name_map={})
    if n_fixed > 0:
        print(f"🩹 Externals reconciled from local upload for {n_fixed} match(es).")

    # NAME ALIGNMENT + DATA-QUALITY FILTERING
    model_players = set(players.keys())
    odds_players  = set(odds["p1_name"].unique()) | set(odds["p2_name"].unique())
    found         = odds_players & model_players
    missing_names = odds_players - model_players
    print("\n🔎 Name alignment report")
    denom = max(1, len(odds_players))
    print(f"  • Players in saved model: {len(model_players):,}")
    print(f"  • Unique names in odds (canonical): {len(odds_players):,}")
    print(f"  • Found in model: {len(found)}  ({(len(found)/denom)*100:.1f}%)")
    print(f"  • Missing in model: {len(missing_names)}  ({(len(missing_names)/denom)*100:.1f}%)")
    if missing_names:
        print("  ⚠️ Missing players (not in model):")
        for name in sorted(missing_names)[:20]:
            print(f"    - {name}")
        if len(missing_names) > 20:
            print(f"    … and {len(missing_names)-20} more.")
    def _player_ok(name, srf):
        P = players.get(name)
        if P is None:
            return False, "missing_in_model"
        tot = len(P["wins_all"])
        surf_n = len(P["wins_surf"].get(srf, deque()))
        if tot < MIN_MATCHES_TOTAL:
            return False, f"low_total_hist({tot})"
        if surf_n < MIN_MATCHES_SURF:
            return False, f"low_surface_hist({surf_n})"
        return True, ""
    if ENFORCE_DATA_QUALITY:
        dropped = []
        keep_idx = []
        for i, r in odds.iterrows():
            srf = r["surface"]
            ok1, why1 = _player_ok(r["p1_name"], srf)
            ok2, why2 = _player_ok(r["p2_name"], srf)
            if ok1 and ok2:
                keep_idx.append(i)
            else:
                reason = []
                if not ok1: reason.append(f"{r['p1_name']}: {why1}")
                if not ok2: reason.append(f"{r['p2_name']}: {why2}")
                dropped.append((r["tourney_name"], r["p1_name"], r["p2_name"], srf, "; ".join(reason)))
        if dropped:
            print("\n🚫 Auto-dropped matches due to data sufficiency:")
            for tn, a, b, s, why in dropped[:10]:
                print(f"  - {tn} [{s}]: {a} vs {b}  →  {why}")
            if len(dropped) > 10:
                print(f"  … and {len(dropped)-10} more dropped.")
        odds = odds.loc[keep_idx].reset_index(drop=True)
        if odds.empty:
            print("\n❌ No matches left after data-quality filtering. Exiting.")
            return
    # Build features row-by-row
    X_rows = []; surf_vec = []
    dl_inputs_p1 = []; dl_inputs_p2 = []
    def winrate(dq): return (sum(dq)/len(dq)) if len(dq) else 0.0
    for _, r in odds.iterrows():
        p1 = r["p1_name"]; p2 = r["p2_name"]; srf = r["surface"]; now = pd.to_datetime(r["event_date"])
        P1 = players.get(p1); P2 = players.get(p2)
        feat = {
            "Year": now.year, "Date": now, "surface_label": srf,
            "p1_name": p1, "p2_name": p2,
            "elo_all_diff":       P1["elo_all"] - P2["elo_all"],
            "elo_surf_diff":      P1["elo_surf"].get(srf,1500.0) - P2["elo_surf"].get(srf,1500.0),
            "elo_sets_all_diff": P1["elo_sets_all"] - P2["elo_sets_all"],
            "elo_sets_surf_diff":P1["elo_sets_surf"].get(srf,1500.0) - P2["elo_sets_surf"].get(srf,1500.0),
            "winrate_all_diff":  winrate(P1["wins_all"]) - winrate(P2["wins_all"]),
            "winrate_surf_diff": winrate(P1["wins_surf"].get(srf,deque())) - winrate(P2["wins_surf"].get(srf,deque())),
            "matches_all_diff":  len(P1["wins_all"]) - len(P2["wins_all"]),
            "matches_surf_diff": len(P1["wins_surf"].get(srf,deque())) - len(P2["wins_surf"].get(srf,deque())),
            "days_since_last_surface_diff": last_surface_gap_days(P1, srf, now, clip=CLIP_DAYS_ABS) - last_surface_gap_days(P2, srf, now, clip=CLIP_DAYS_ABS),
            "long_layoff_flag_diff": (1 if (P1["recent_dates"] and (now - P1["recent_dates"][-1]).days > CLIP_DAYS_ABS) else 0) - \
                                     (1 if (P2["recent_dates"] and (now - P2["recent_dates"][-1]).days > CLIP_DAYS_ABS) else 0),
            "matches_7d_diff": recent_matches_7d(P1, now) - recent_matches_7d(P2, now),
            "surface_switch3_diff": surf_share3(P1, srf) - surf_share3(P2, srf),
            "tier_elo_diff": P1["tier_elo"]["GS"] - P2["tier_elo"]["GS"],
            "h2h_norm": 0.0, "h2h_total_capped": 0
        }
        pair = tuple(sorted([p1,p2])); p1_h2h = h2h.get(pair,{}).get(p1,0); p2_h2h = h2h.get(pair,{}).get(p2,0)
        tot = p1_h2h+p2_h2h
        feat["h2h_norm"] = (p1_h2h - p2_h2h)/max(1,tot)
        feat["h2h_total_capped"] = min(8, tot)
        X_rows.append(feat)
        surf_vec.append(srf)
        def seq_from_player_now(P, srf):
            seq = list(P["seq_hist"])
            if not seq:
                return np.zeros((roll_n, len(cont_names)), dtype=np.float32)
            arr = np.stack(seq[-roll_n:], axis=0)
            if arr.shape[0] < roll_n:
                pad = np.zeros((roll_n - arr.shape[0], arr.shape[1]), dtype=np.float32)
                arr = np.vstack([pad, arr])
            return arr.astype(np.float32)
        dl_inputs_p1.append(seq_from_player_now(P1, srf))
        dl_inputs_p2.append(seq_from_player_now(P2, srf))
    X = pd.DataFrame(X_rows).fillna(0.0).replace([np.inf,-np.inf], 0.0)
    for c in kept_feats:
        if c not in X.columns and c != "dl_pred":
            X[c] = 0.0
    p1_seq = np.stack(dl_inputs_p1, axis=0).astype(np.float32)
    p2_seq = np.stack(dl_inputs_p2, axis=0).astype(np.float32)
    dl_pred = call_savedmodel_flex(serve, p1_seq, p2_seq)
    X["dl_pred"] = np.clip(dl_pred, 1e-4, 1-1e-4)
    X_use = X[[c for c in kept_feats]]  # preserves order
    raw = stack.predict_proba(X_use.values)[:,1]
    cal = apply_calibration(method, params, raw, surface=np.array(surf_vec))
    out = odds.copy()
    out["prob_p1"] = cal
    out["prob_p2"] = 1 - out["prob_p1"]
    # Keep a copy of original main model probs as OP for later (SG-as-main plan)
    out["op_p1"] = out["prob_p1"]
    out["op_p2"] = out["prob_p2"]
    # === Blend externals into probabilities (tempered logit) === (Main model path only)
    if USE_EXTERNAL_SIGNALS and USE_EXTERNAL_IN_PROB:
        def _logit(p): return np.log(_clip01(p)) - np.log(_clip01(1-p))
        def _sigm(x): return 1.0 / (1.0 + np.exp(-x))
        pm = out["prob_p1"].values
        pe = out["ext_p1_agg"].values
        mask = ~np.isnan(pe)
        if mask.any():
            lm = _logit(pm[mask])
            le = _logit(pe[mask]) / float(EXT_TEMP)
            w_m = float(EXT_BLEND_W_MODEL); w_e = 1.0 - w_m
            lb = w_m*lm + w_e*le
            out.loc[mask, "prob_p1"] = _clip01(_sigm(lb))
        out["prob_p2"] = 1 - out["prob_p1"]
    # Book EV + implied (based on main-model probs)
    out["ev_book_p1"] = out["prob_p1"] * out["p1_odds"] - 1
    out["ev_book_p2"] = out["prob_p2"] * out["p2_odds"] - 1
    with np.errstate(divide="ignore", invalid="ignore"):
        out["imp_book_p1_raw"] = 1.0 / out["p1_odds"]
        out["imp_book_p2_raw"] = 1.0 / out["p2_odds"]
    s = out["imp_book_p1_raw"] + out["imp_book_p2_raw"]
    out["imp_book_p1"] = out["imp_book_p1_raw"] / s
    out["imp_book_p2"] = out["imp_book_p2_raw"] / s
    out["edge_book_p1"] = out["prob_p1"] - out["imp_book_p1"]
    out["edge_book_p2"] = out["prob_p2"] - out["imp_book_p2"]
    print("📄 Reading Exchange snapshot (if exists)…")
    exch = load_exchange_snapshot_csv(EXCH_SNAPSHOT)
    if not exch.empty:
        if "master_home" in exch.columns: exch["master_home"] = exch["master_home"].map(lambda x: resolve_name(x, name_map))
        if "master_away" in exch.columns: exch["master_away"] = exch["master_away"].map(lambda x: resolve_name(x, name_map))
    pred = add_exchange_to_preds(out, exch)
    # External EVs vs chosen odds (diagnostics)
    if USE_EXTERNAL_SIGNALS:
        with np.errstate(invalid='ignore'):
            for src in ["gemini","tennisabstract","steveg"]:
                pred[f"{src}_ev_p1"] = pred[f"ext_{src}_p1"] * pred["best_odds_p1"] - 1
                pred[f"{src}_ev_p2"] = pred[f"ext_{src}_p2"] * pred["best_odds_p2"] - 1
    # Save combined CSV
    try:
        pred.to_csv("/content/prediction_features.csv", index=False)
        print("💾 Saved '/content/prediction_features.csv'")
    except Exception as e:
        print("⚠️ Could not save combined CSV:", e)
    # ── Selected Bets #1: Main model (yours) with externals SG/G/TA
    print("💰 Building betting plan (Main model, EV-first, consensus & variance-aware)…")
    plan_main, bankroll, exp_profit_main = build_betting_plan_generic(
        pred.copy(), BANKROLL_DKK,
        prob_cols=("prob_p1","prob_p2"),
        ext_sources=EXT_SOURCES_BASE
    )
    if plan_main.empty:
        print("No qualifying bets under current filters (Main).")
    else:
        total_stake = plan_main["stake_dkk"].sum()
        print(f"🧾 Selected bets (MAIN): {len(plan_main)} | Total stake: {total_stake:,.0f} DKK (cap {TOTAL_RISK_CAP_PCT:.0%} → {BANKROLL_DKK*TOTAL_RISK_CAP_PCT:,.0f} DKK)")
        print(f"   Expected profit (batch): {exp_profit_main:+.0f} DKK")
        print("\n── Selected Bets (MAIN — blended prob + exchange needed-odds @ 6.5% commission) ──")
        for _, r in plan_main.iterrows():
            src_txt = "Betfair Exchange" if r["source"]=="EXCH" else "Book"
            matchup = f"{r['p1_name']} vs {r['p2_name']}"
            pick_txt = f"{r['player']} to Win"
            actual_exch_txt = f"{r['exch_actual_side']:.2f}" if pd.notna(r["exch_actual_side"]) else "—"
            origin = ""
            if "externals_source" in r and isinstance(r["externals_source"], str):
                origin = " [externals=UPLOAD]"
            print(
                f"- {r['tourney_name']} | {matchup} | {pick_txt} | {src_txt} @ {r['odds']:.2f} | "
                f"p_blend={r['blend_prob']:.1%} | value_agree={r['ext_agree_value']}/3 {r['ext_support']} | "
                f"stake {r['stake_dkk']:,.0f} DKK | EV {r['ev_decimal']:.1%} | "
                f"book={r['book_odds_side']:.2f} → needed exch={r['exch_needed_side']:.2f}, actual exch={actual_exch_txt}"
                f"{origin}"
            )
        plan_main.to_csv("/content/betting_plan_main.csv", index=False)
        print("💾 Saved '/content/betting_plan_main.csv'")
    # ── Selected Bets #2: SG-as-Main (swap) — SG main probs; externals are G, TA, and OP(=your old main)
    # Prepare SG main probability cols; if SG missing → drop those rows
    pred_swap = pred.copy()
    pred_swap["sg_p1"] = pred_swap["ext_steveg_p1"]
    pred_swap["sg_p2"] = pred_swap["ext_steveg_p2"]
    # attach OP probs already saved as op_p1/op_p2; make them "external" for swap logic
    # (we already have pred_swap["op_p1"], pred_swap["op_p2"] from earlier)
    # Only keep rows where SG has both sides
    pred_swap = pred_swap[ pred_swap["sg_p1"].notna() & pred_swap["sg_p2"].notna() ].reset_index(drop=True)
    if pred_swap.empty:
        print("\nNo matches have SG probabilities, skipping SG-as-Main section.")
        return
    print("\n💰 Building betting plan (SG-as-Main, externals = G/TA/OP)…")
    plan_sg, _, exp_profit_sg = build_betting_plan_generic(
        pred_swap, BANKROLL_DKK,
        prob_cols=("sg_p1","sg_p2"),
        ext_sources=EXT_SOURCES_SWAP
    )
    if plan_sg.empty:
        print("No qualifying bets under current filters (SG-as-Main).")
    else:
        total_stake = plan_sg["stake_dkk"].sum()
        print(f"🧾 Selected bets (SG-as-Main): {len(plan_sg)} | Total stake: {total_stake:,.0f} DKK (cap {TOTAL_RISK_CAP_PCT:.0%} → {BANKROLL_DKK*TOTAL_RISK_CAP_PCT:,.0f} DKK)")
        print(f"   Expected profit (batch): {exp_profit_sg:+.0f} DKK")
        print("\n── Selected Bets (SG-as-Main — SG prob + exchange needed-odds @ 6.5% commission) ──")
        for _, r in plan_sg.iterrows():
            src_txt = "Betfair Exchange" if r["source"]=="EXCH" else "Book"
            matchup = f"{r['p1_name']} vs {r['p2_name']}"
            pick_txt = f"{r['player']} to Win"
            actual_exch_txt = f"{r['exch_actual_side']:.2f}" if pd.notna(r["exch_actual_side"]) else "—"
            origin = ""
            if "externals_source" in r and isinstance(r["externals_source"], str):
                origin = " [externals=UPLOAD]"
            print(
                f"- {r['tourney_name']} | {matchup} | {pick_txt} | {src_txt} @ {r['odds']:.2f} | "
                f"p_SG={r['blend_prob']:.1%} | value_agree={r['ext_agree_value']}/3 {r['ext_support']} | "
                f"stake {r['stake_dkk']:,.0f} DKK | EV {r['ev_decimal']:.1%} | "
                f"book={r['book_odds_side']:.2f} → needed exch={r['exch_needed_side']:.2f}, actual exch={actual_exch_txt}"
                f"{origin}"
            )
        plan_sg.to_csv("/content/betting_plan_sg_main.csv", index=False)
        print("💾 Saved '/content/betting_plan_sg_main.csv'")

# Run
run()
