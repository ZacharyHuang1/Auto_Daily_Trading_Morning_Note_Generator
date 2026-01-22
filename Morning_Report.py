from __future__ import annotations

import os
import sys
import traceback
import io
import base64
import math
import re
import html as html_lib
import webbrowser
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

import pandas as pd

import matplotlib
matplotlib.use("Agg")  # headless render
import matplotlib.pyplot as plt


# ----------------------------
# Utilities
# ----------------------------

def load_dotenv(path: str = ".env") -> None:
    """
    Minimal .env loader (no extra dependency).
    Supports lines like KEY=VALUE, ignores comments and empty lines.
    """
    p = Path(path)
    if not p.exists():
        return
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        if k and k not in os.environ:
            os.environ[k] = v


def is_nan(x) -> bool:
    try:
        return x is None or (isinstance(x, float) and math.isnan(x))
    except Exception:
        return False


def fmt_num(x, nd=2, suffix=""):
    if is_nan(x):
        return "N/A"
    try:
        return f"{float(x):.{nd}f}{suffix}"
    except Exception:
        return "N/A"


def fmt_signed(x, nd=2, suffix=""):
    if is_nan(x):
        return "N/A"
    try:
        val = float(x)
        sign = "+" if val > 0 else ""
        return f"{sign}{val:.{nd}f}{suffix}"
    except Exception:
        return "N/A"


def fig_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


COLORIZE_COL_KEYWORDS = ("Chg", "Δ", "Change", "Surprise", "Spread", "Return", "Ret")


def colorize_value(val, col_name: str = "") -> str:
    if is_nan(val):
        return "N/A"
    col_lower = col_name.lower() if col_name else ""
    if isinstance(val, str):
        s = val.strip()
        if s in ("", "N/A"):
            return s
        if s.startswith("<"):
            return s
        is_pct = "%" in s or "pct" in col_lower or "chg%" in col_lower or "return" in col_lower
        raw = s.replace("%", "").replace("bp", "").replace(",", "")
        raw = raw.replace("pips", "").replace("PIPS", "").strip()
        try:
            num = float(raw)
        except Exception:
            return s
        if num > 0:
            return f"<span class='pos'>{s}</span>"
        if num < 0:
            return f"<span class='neg'>{s}</span>"
        return s
    try:
        num = float(val)
    except Exception:
        return str(val)
    is_pct = "%" in col_name or "pct" in col_lower or "chg%" in col_lower or "return" in col_lower
    if num > 0:
        return f"<span class='pos'>{val}</span>"
    if num < 0:
        return f"<span class='neg'>{val}</span>"
    return str(val)


def df_to_html_table(df: pd.DataFrame, colorize_cols: Optional[List[str]] = None) -> str:
    if df is None or df.empty:
        return "<div class='muted'>No data.</div>"
    out = df.copy()
    if colorize_cols is None:
        colorize_cols = [c for c in out.columns if any(k in str(c) for k in COLORIZE_COL_KEYWORDS)]
    for c in colorize_cols:
        if c in out.columns:
            out[c] = out[c].map(lambda v: colorize_value(v, str(c)))
    return out.to_html(index=False, border=0, classes="tbl", justify="left", escape=False)


def mark_stale_dates(
    df: pd.DataFrame,
    date_col: str = "Date",
    max_age_days: int = 45,
    label: str = "Delayed Data",
) -> pd.DataFrame:
    if df is None or df.empty or date_col not in df.columns:
        return df
    out = df.copy()
    today = datetime.now().date()

    def mark(val):
        try:
            d = pd.to_datetime(val).date()
        except Exception:
            return val
        age = (today - d).days
        if age > max_age_days:
            return f"{d.isoformat()} <span class='stale'>({label})</span>"
        return d.isoformat()

    out[date_col] = out[date_col].map(mark)
    return out


def mark_delayed_if_missing(
    df: pd.DataFrame,
    date_col: str = "Date",
    actual_col: str = "Actual",
    label: str = "Delayed Data",
) -> pd.DataFrame:
    if df is None or df.empty or date_col not in df.columns or actual_col not in df.columns:
        return df
    out = df.copy()

    def mark(row):
        val = row.get(actual_col)
        if is_nan(val) or str(val).strip() in ("", "N/A"):
            return f"{row.get(date_col, '')} <span class='stale'>({label})</span>"
        return row.get(date_col, "")

    out[date_col] = out.apply(mark, axis=1)
    return out


def contains_cjk(text: str) -> bool:
    for ch in text:
        if "\u4e00" <= ch <= "\u9fff":
            return True
    return False


def translate_region(region: str) -> str:
    mapping = {
        "中国": "China",
        "美国": "US",
        "日本": "Japan",
        "英国": "UK",
        "欧元区": "Eurozone",
        "德国": "Germany",
        "法国": "France",
        "韩国": "Korea",
        "澳大利亚": "Australia",
        "新西兰": "New Zealand",
        "加拿大": "Canada",
        "香港": "Hong Kong",
        "瑞士": "Switzerland",
    }
    return mapping.get(region, region)


def translate_event_text(event: str) -> str:
    text = str(event)
    if not contains_cjk(text):
        return text

    month_map = {
        "12月": "Dec",
        "11月": "Nov",
        "10月": "Oct",
        "9月": "Sep",
        "8月": "Aug",
        "7月": "Jul",
        "6月": "Jun",
        "5月": "May",
        "4月": "Apr",
        "3月": "Mar",
        "2月": "Feb",
        "1月": "Jan",
    }
    for k, v in month_map.items():
        text = text.replace(k, v)

    replacements = {
        "一年期贷款市场报价利率": "1Y LPR",
        "五年期贷款市场报价利率": "5Y LPR",
        "贷款市场报价利率": "LPR",
        "包括红利三个月平均工资年率": "3M avg earnings YoY (incl bonus)",
        "失业率-按ILO标准": "Unemployment rate (ILO)",
        "商品贸易帐": "Goods trade balance",
        "商品出口": "Goods exports",
        "商品进口": "Goods imports",
        "贸易帐": "Trade balance",
        "经常帐": "Current account",
        "就业人口变动": "Employment change",
        "就业人口": "Employment",
        "失业率": "Unemployment rate",
        "就业": "Employment",
        "出口": "Exports",
        "进口": "Imports",
        "每日仓单变动-铜": "daily warrant change - Copper",
        "每日": "daily",
        "上期所": "SHFE",
        "SHFEdaily": "SHFE daily",
        "COMEXGold": "COMEX Gold",
        "SPDRGold": "SPDR Gold",
        "iSharesGold": "iShares Gold",
        "iSharesSilver": "iShares Silver",
        "黄金": "Gold",
        "白银": "Silver",
        "铜": "Copper",
        "库存": "Inventories",
        "持仓": "Holdings",
        "更新": "update",
        "dailyupdate": "daily update",
        "核心": "Core",
        "核心CPI": "Core CPI",
        "零售物价指数": "RPI",
        "物价指数": "Price Index",
        "零售": "Retail",
        "输入PPI": "Input PPI",
        "未季调": "NSA",
        "季调后": "SA",
        "季调": "SA",
        "CBI企业乐观指数": "CBI business optimism",
        "企业乐观指数": "CBI business optimism",
        "第一季度": "Q1",
        "第二季度": "Q2",
        "第三季度": "Q3",
        "第四季度": "Q4",
        "中国": "China",
        "英国": "UK",
        "美国": "US",
        "欧元区": "Eurozone",
        "日本": "Japan",
        "瑞士": "Switzerland",
        "澳大利亚": "Australia",
        "新西兰": "New Zealand",
        "韩国": "Korea",
        "同比": "YoY",
        "环比": "MoM",
        "年率": "YoY",
        "月率": "MoM",
        "周": "Week",
        "吨": "tons",
        "盎司": "oz",
        "万人": "10k",
        "亿日元": "JPY 100m",
        "百": "100 ",
    }
    for k, v in replacements.items():
        text = text.replace(k, v)

    text = re.sub(r"截至\s*(\d+月\d+日)当周", r"Week of \1", text)
    text = re.sub(r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)(\d+)日", r"\1 \2", text)
    text = re.sub(r"([A-Za-z])(?=(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec))", r"\1 ", text)
    text = re.sub(r"([A-Za-z])(?=Q[1-4])", r"\1 ", text)
    text = re.sub(r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s*(\d+)", r"\1 \2", text)
    text = re.sub(r"(\d)([A-Za-z])", r"\1 \2", text)
    text = re.sub(r"([A-Z]{2,})([A-Z][a-z])", r"\1 \2", text)
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    text = re.sub(r"\s*-\s*", " - ", text)
    text = text.replace("CBICBI", "CBI")
    text = text.replace("i Shares", "iShares")
    text = re.sub(r"\((\S)", r"( \1", text)
    text = re.sub(r"\(\s*%\s*\)", "(%)", text)
    text = text.replace("( 100 oz)", "(100 oz)")
    text = text.replace("100 m", "100m").replace("10 k", "10k")
    text = re.sub(r"\(\s+", "(", text)
    text = re.sub(r"\s+\)", ")", text)
    text = text.replace("Yo Y", "YoY").replace("Mo M", "MoM")
    text = text.replace("截至", "Week of ")
    return text


# ----------------------------
# AKShare Fetchers
# ----------------------------

def fetch_us_indices() -> Tuple[pd.DataFrame, Dict[str, pd.Series], str]:
    """
    US indices from Sina via AKShare:
    symbols: .INX (S&P 500), .IXIC (NASDAQ Composite), .DJI (Dow), .NDX (Nasdaq 100)
    Returns:
      snapshot_df: table with close/chg/chg%
      series_map: close series for charting
    """
    import akshare as ak

    idx_map = {
        ".INX": "S&P 500",
        ".IXIC": "NASDAQ Comp",
        ".DJI": "Dow Jones",
        ".NDX": "Nasdaq 100",
    }

    rows = []
    series_map: Dict[str, pd.Series] = {}
    latest_dates: List[datetime.date] = []

    for symbol, name in idx_map.items():
        try:
            df = ak.index_us_stock_sina(symbol=symbol)
            if df is None or df.empty:
                continue
            df = df.copy()
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date")
            # last two trading days
            last = df.iloc[-1]
            prev = df.iloc[-2] if len(df) >= 2 else df.iloc[-1]
            close = float(last["close"])
            prev_close = float(prev["close"])
            chg = close - prev_close
            chg_pct = (chg / prev_close * 100.0) if prev_close else float("nan")
            last_date = last["date"].date()
            rows.append({
                "Asset": name,
                "Date": last_date.isoformat(),
                "Close": fmt_num(close, 2),
                "Chg": fmt_signed(chg, 2),
                "Chg%": fmt_signed(chg_pct, 2, "%"),
            })
            latest_dates.append(last_date)

            # keep last ~120 sessions for chart
            tail = df.tail(160).set_index("date")["close"].astype(float)
            series_map[name] = tail
        except Exception as e:
            rows.append({
                "Asset": name,
                "Date": "N/A",
                "Close": "N/A",
                "Chg": "N/A",
                "Chg%": "N/A",
            })

    snapshot_df = pd.DataFrame(rows)

    market_note = ""
    if latest_dates:
        latest_date = max(latest_dates)
        today = datetime.now().date()
        is_trading_day = None
        try:
            import pandas_market_calendars as mcal
            nyse = mcal.get_calendar("NYSE")
            schedule = nyse.schedule(start_date=today, end_date=today)
            is_trading_day = not schedule.empty
        except Exception:
            is_trading_day = today.weekday() < 5
        if not is_trading_day and latest_date < today:
            market_note = "Market Closed - Holiday"
        elif is_trading_day and latest_date < today:
            market_note = "Market Data Delayed"

    return snapshot_df, series_map, market_note


def plot_us_indices_normalized(series_map: Dict[str, pd.Series]) -> Optional[str]:
    if not series_map:
        return None

    # align by date index union
    df = pd.DataFrame({k: v for k, v in series_map.items()}).sort_index()

    # normalize to 100 at first available point per series
    norm = df.copy()
    for col in norm.columns:
        s = norm[col].dropna()
        if s.empty:
            continue
        base = s.iloc[0]
        norm[col] = norm[col] / base * 100.0

    fig = plt.figure(figsize=(10.5, 4.2))
    ax = fig.add_subplot(111)
    for col in norm.columns:
        ax.plot(norm.index, norm[col], label=col)
    ax.set_title("US Indices (Normalized to 100)")
    ax.set_ylabel("Index (normalized)")
    ax.grid(True, alpha=0.2)
    ax.legend(loc="best", fontsize=8)
    return fig_to_base64(fig)


def tenor_to_years(tenor: str) -> float:
    """Convert tenor label like '3M'/'10Y'/'1.5M'/'6W' into year fraction.

    Used for sorting/plotting yield curves.
    """
    t = str(tenor).strip().upper()
    m = re.match(r"^(\d+(?:\.\d+)?)([MYW])$", t)
    if not m:
        return float('inf')
    n = float(m.group(1))
    unit = m.group(2)
    if unit == 'W':
        return n / 52.0
    if unit == 'M':
        return n / 12.0
    return float(n)


def fetch_ust_curve_from_treasury(start_date: datetime) -> pd.DataFrame:
    """Fetch Daily Treasury Yield Curve Rates from home.treasury.gov (official CSV dataset).

    Returns a DataFrame with columns like:
      Date, 1M, 1.5M (if available), 2M, 3M, 4M, 6M, 1Y, 2Y, 3Y, 5Y, 7Y, 10Y, 20Y, 30Y

    Notes:
      - Uses the Treasury CSV endpoint (more robust than scraping HTML).
      - Falls back to empty DataFrame on any failure.
    """

    def _fmt_num(n: float) -> str:
        # avoid 1.0 -> '1'
        if abs(n - int(n)) < 1e-9:
            return str(int(n))
        return str(n).rstrip('0').rstrip('.')

    def _normalize_treasury_tenor(col: str) -> Optional[str]:
        s = str(col).strip().replace(" ", " ")
        s = re.sub(r"\s+", " ", s)
        s_low = s.lower().strip('"').strip("'")

        # Examples: '1 Mo', '1.5 Mo', '10 Yr'
        m = re.match(r"^(\d+(?:\.\d+)?)\s*mo$", s_low)
        if m:
            n = float(m.group(1))
            return f"{_fmt_num(n)}M"

        m = re.match(r"^(\d+(?:\.\d+)?)\s*yr$", s_low)
        if m:
            n = float(m.group(1))
            return f"{_fmt_num(n)}Y"

        # Slightly more verbose variants
        m = re.match(r"^(\d+(?:\.\d+)?)\s*(month|months)$", s_low)
        if m:
            n = float(m.group(1))
            return f"{_fmt_num(n)}M"

        m = re.match(r"^(\d+(?:\.\d+)?)\s*(year|years)$", s_low)
        if m:
            n = float(m.group(1))
            return f"{_fmt_num(n)}Y"

        return None

    def _treasury_read_csv(url: str) -> pd.DataFrame:
        """Read Treasury CSV with a browser-like UA (some endpoints block default urllib)."""
        # requests is preferred but optional
        try:
            import requests  # type: ignore

            resp = requests.get(
                url,
                headers={"User-Agent": "Mozilla/5.0"},
                timeout=20,
            )
            resp.raise_for_status()
            txt = resp.text
            if "<html" in txt.lower():
                raise ValueError("Unexpected HTML response")
            return pd.read_csv(io.StringIO(txt))
        except Exception:
            try:
                return pd.read_csv(url)
            except Exception:
                return pd.DataFrame()

    try:
        current_year = datetime.now(timezone.utc).year
        years = list(range(start_date.year, current_year + 1))
        dfs: List[pd.DataFrame] = []

        for y in years:
            url = (
                "https://home.treasury.gov/resource-center/data-chart-center/interest-rates/"
                f"daily-treasury-rates.csv/{y}/all"
                f"?_format=csv&field_tdr_date_value={y}&page=&type=daily_treasury_yield_curve"
            )

            df = _treasury_read_csv(url)
            if df is None or df.empty:
                continue

            df = df.copy()
            df = df.loc[:, ~df.columns.duplicated()]

            # normalize date column
            date_col = None
            for c in df.columns:
                if str(c).strip().lower() in ("date", "日期"):
                    date_col = c
                    break
            if date_col is None:
                date_col = df.columns[0]
            df.rename(columns={date_col: "Date"}, inplace=True)

            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

            # normalize tenor columns
            rename: Dict[str, str] = {}
            for c in df.columns:
                if c == "Date":
                    continue
                t = _normalize_treasury_tenor(c)
                if t:
                    rename[c] = t

            if rename:
                df.rename(columns=rename, inplace=True)

            tenor_cols = sorted(set(rename.values()), key=tenor_to_years)
            keep_cols = ["Date"] + [c for c in tenor_cols if c in df.columns]
            df = df[keep_cols].copy() if keep_cols else df[["Date"]].copy()

            for c in df.columns:
                if c == "Date":
                    continue
                df[c] = pd.to_numeric(df[c], errors="coerce")

            df = df.dropna(subset=["Date"]).sort_values("Date")
            dfs.append(df)

        if not dfs:
            return pd.DataFrame()

        out = pd.concat(dfs, ignore_index=True)
        out = out.loc[:, ~out.columns.duplicated()]
        out = out.drop_duplicates(subset=["Date"], keep="last")
        out = out[out["Date"] >= start_date].sort_values("Date")

        # scale check: if yields look like decimals (0.04), convert to percent
        y_cols = [c for c in out.columns if c != "Date"]
        if y_cols:
            sample = out[y_cols].stack().dropna()
            if not sample.empty and sample.median() < 1.0:
                out[y_cols] = out[y_cols] * 100.0

        return out.reset_index(drop=True)
    except Exception:
        return pd.DataFrame()


def fetch_ust_curve_from_akshare(days_back: int = 420) -> pd.DataFrame:
    """Fallback UST yields from AKShare (EastMoney).

    Standardizes columns into Date + tenor labels like 2Y/5Y/10Y/30Y if present.
    """
    try:
        import akshare as ak
    except Exception:
        return pd.DataFrame()

    try:
        df = ak.bond_zh_us_rate()
    except Exception:
        return pd.DataFrame()

    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()
    # normalize date
    if "日期" in df.columns:
        df.rename(columns={"日期": "Date"}, inplace=True)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    # map tenor columns
    rename = {}
    for c in df.columns:
        if c == "Date":
            continue
        c0 = str(c)
        # e.g. 美国国债收益率10年, 美国国债收益率3月
        m = re.search(r"美国国债收益率\s*(\d+)\s*(年|月)", c0)
        if m:
            n = int(m.group(1))
            unit = m.group(2)
            rename[c] = f"{n}{'Y' if unit == '年' else 'M'}"

    if rename:
        df.rename(columns=rename, inplace=True)

    keep = ["Date"] + [v for v in rename.values() if v in df.columns]
    out = df[keep].copy() if keep else df[["Date"]].copy()

    # numeric
    for c in out.columns:
        if c == "Date":
            continue
        out[c] = pd.to_numeric(out[c], errors="coerce")

    out = out.dropna(subset=["Date"]).sort_values("Date")
    if days_back and len(out) > days_back:
        out = out.tail(days_back)

    # scale check: if yields look like decimals, convert to percent
    y_cols = [c for c in out.columns if c != "Date"]
    if y_cols:
        sample = out[y_cols].stack().dropna()
        if not sample.empty and sample.median() < 1.0:
            out[y_cols] = out[y_cols] * 100.0

    return out.reset_index(drop=True)


def last_two_non_nan(s: pd.Series) -> Tuple[float, float]:
    """Return (latest, previous) for a numeric series, skipping NaNs."""
    s2 = pd.to_numeric(s, errors="coerce").dropna()
    if len(s2) == 0:
        return float('nan'), float('nan')
    if len(s2) == 1:
        return float(s2.iloc[-1]), float('nan')
    return float(s2.iloc[-1]), float(s2.iloc[-2])


def compute_key_spreads_bp(hist: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """Compute key curve spreads in **bp** from the standardized hist curve."""
    out: Dict[str, Dict[str, float]] = {}
    if hist is None or hist.empty:
        return out

    def spread(long_t: str, short_t: str) -> Dict[str, float]:
        if long_t not in hist.columns or short_t not in hist.columns:
            return {"level_bp": float('nan'), "chg_bp": float('nan')}
        l_last, l_prev = last_two_non_nan(hist[long_t])
        s_last, s_prev = last_two_non_nan(hist[short_t])
        if is_nan(l_last) or is_nan(s_last):
            return {"level_bp": float('nan'), "chg_bp": float('nan')}
        level = (l_last - s_last) * 100.0
        chg = float('nan')
        if not is_nan(l_prev) and not is_nan(s_prev):
            prev_level = (l_prev - s_prev) * 100.0
            chg = level - prev_level
        return {"level_bp": level, "chg_bp": chg}

    out["2s10s"] = spread("10Y", "2Y")
    out["5s30s"] = spread("30Y", "5Y")  # long - short
    return out


def build_spreads_table(hist: pd.DataFrame) -> pd.DataFrame:
    """Small table for key UST spreads."""
    sp = compute_key_spreads_bp(hist)
    rows = []
    label_map = {
        "2s10s": "2s10s (10Y-2Y)",
        "5s30s": "5s30s (30Y-5Y)",
    }
    for k in ["2s10s", "5s30s"]:
        if k not in sp:
            continue
        level = sp[k]["level_bp"]
        chg = sp[k]["chg_bp"]
        rows.append({
            "Spread": label_map.get(k, k),
            "Level (bp)": fmt_signed(level, 1) if not is_nan(level) else "N/A",
            "Δ (bp)": fmt_signed(chg, 1) if not is_nan(chg) else "",
        })
    return pd.DataFrame(rows)


def fetch_us_treasury_yields(days_back: int = 420) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch UST yield curve history and build latest snapshot.

    Returns:
      latest_table: Tenor / Yield / Δ(bp)
      hist: Date + tenor columns (numeric, %)
    """
    # 1) Try US Treasury official curve (more points)
    start_date = datetime.now(timezone.utc).date() - timedelta(days=int(days_back * 1.2))
    hist = fetch_ust_curve_from_treasury(datetime.combine(start_date, datetime.min.time()))

    # 2) Fallback to AKShare (less points)
    if hist is None or hist.empty:
        hist = fetch_ust_curve_from_akshare(days_back=days_back)

    if hist is None or hist.empty:
        return pd.DataFrame(), pd.DataFrame()

    # ensure Date column is datetime
    hist = hist.copy()
    # guard against duplicate column names (can break pd.to_datetime)
    hist = hist.loc[:, ~hist.columns.duplicated()]

    if "Date" not in hist.columns:
        # try common alternatives
        for c in list(hist.columns):
            if str(c).strip().lower() in ("date", "日期"):
                hist.rename(columns={c: "Date"}, inplace=True)
                break

    date_obj = hist.get("Date")
    # If duplicate 'Date' columns slipped through, pandas returns a DataFrame here.
    if isinstance(date_obj, pd.DataFrame):
        date_obj = date_obj.iloc[:, 0]

    hist["Date"] = pd.to_datetime(date_obj, errors="coerce")
    hist = hist.dropna(subset=["Date"]).sort_values("Date")

    # Build latest snapshot table
    tenor_cols = [c for c in hist.columns if c != "Date"]
    tenor_cols = sorted(tenor_cols, key=tenor_to_years)

    rows = []
    for t in tenor_cols:
        last, prev = last_two_non_nan(hist[t])
        chg_bp = (last - prev) * 100.0 if (not is_nan(last) and not is_nan(prev)) else float('nan')
        rows.append({
            "Tenor": t,
            "Yield": fmt_num(last, 2, "%"),
            "Δ(bp)": fmt_signed(chg_bp, 1) if not is_nan(chg_bp) else "N/A",
        })

    latest_table = pd.DataFrame(rows)
    return latest_table, hist.reset_index(drop=True)


def plot_ust_curve(hist: pd.DataFrame) -> str:
    """Yield curve (latest point)."""
    if hist is None or hist.empty:
        return ""

    tenor_cols = [c for c in hist.columns if c != "Date"]
    tenor_cols = sorted(tenor_cols, key=tenor_to_years)
    latest = hist.dropna(subset=["Date"]).sort_values("Date").iloc[-1]

    xs = [tenor_to_years(t) for t in tenor_cols]
    ys = [float(latest.get(t)) if not is_nan(latest.get(t)) else float('nan') for t in tenor_cols]

    # remove NaNs
    x2, y2, lab = [], [], []
    for x, y, t in zip(xs, ys, tenor_cols):
        if is_nan(y) or is_nan(x) or x == float('inf'):
            continue
        x2.append(x)
        y2.append(y)
        lab.append(t)

    fig, ax = plt.subplots(figsize=(8.2, 3.2))
    ax.plot(x2, y2, marker="o")
    ax.set_title("UST Yield Curve")
    ax.set_xlabel("Maturity (Years)")
    ax.set_ylabel("Yield (%)")
    ax.grid(True, alpha=0.3)
    # annotate a few key points
    for x, y, t in zip(x2, y2, lab):
        if t in ("3M", "2Y", "5Y", "10Y", "30Y"):
            ax.annotate(f"{t}\n{y:.2f}%", (x, y), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8)
    return fig_to_base64(fig)


def plot_ust_curve_compare(hist: pd.DataFrame, lookback_points: int = 5) -> str:
    """Compare latest curve vs ~1W ago (lookback_points trading days)."""
    if hist is None or hist.empty:
        return ""

    df = hist.dropna(subset=["Date"]).sort_values("Date")
    if len(df) < lookback_points + 1:
        return ""

    latest = df.iloc[-1]
    prev = df.iloc[-(lookback_points + 1)]

    tenor_cols = [c for c in df.columns if c != "Date"]
    tenor_cols = sorted(tenor_cols, key=tenor_to_years)

    xs = [tenor_to_years(t) for t in tenor_cols]
    y_latest = [float(latest.get(t)) if not is_nan(latest.get(t)) else float('nan') for t in tenor_cols]
    y_prev = [float(prev.get(t)) if not is_nan(prev.get(t)) else float('nan') for t in tenor_cols]

    # Keep points where both exist
    x2, yl2, yp2 = [], [], []
    for x, a, b in zip(xs, y_latest, y_prev):
        if is_nan(x) or x == float('inf') or is_nan(a) or is_nan(b):
            continue
        x2.append(x)
        yl2.append(a)
        yp2.append(b)

    fig, ax = plt.subplots(figsize=(8.2, 3.2))
    ax.plot(x2, yp2, marker="o", label=f"~{lookback_points}d ago")
    ax.plot(x2, yl2, marker="o", label="Latest")
    ax.set_title("UST Yield Curve: Latest vs ~1W")
    ax.set_xlabel("Maturity (Years)")
    ax.set_ylabel("Yield (%)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    return fig_to_base64(fig)


def plot_curve_spread(hist: pd.DataFrame, long_t: str, short_t: str, title: str) -> str:
    """Generic curve spread plot in bp."""
    if hist is None or hist.empty:
        return ""
    df = hist.dropna(subset=["Date"]).sort_values("Date")
    if long_t not in df.columns or short_t not in df.columns:
        return ""

    spread_bp = (df[long_t] - df[short_t]) * 100.0
    fig, ax = plt.subplots(figsize=(8.2, 3.0))
    ax.plot(df["Date"], spread_bp)
    ax.axhline(0, color="grey", linewidth=0.8)
    ax.set_title(title)
    ax.set_ylabel("Spread (bp)")
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    return fig_to_base64(fig)


def plot_2s10s_spread(hist: pd.DataFrame) -> str:
    return plot_curve_spread(hist, "10Y", "2Y", "UST 2s10s Spread")


def plot_5s30s_spread(hist: pd.DataFrame) -> str:
    return plot_curve_spread(hist, "30Y", "5Y", "UST 5s30s Spread")


def build_rates_commodities_table(ust_latest_df: pd.DataFrame, comm_df: pd.DataFrame, ust_hist: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Build small rates & commodities panel.

    Includes US 2Y/10Y yields, key curve spreads (bp), and Gold/Oil.
    """
    rows = []

    # helper: get latest yield + chg from latest table
    def add_yield_row(tenor: str, label: str):
        if ust_latest_df is None or ust_latest_df.empty:
            return
        hit = ust_latest_df[ust_latest_df["Tenor"].astype(str).str.upper() == tenor.upper()]
        if hit.empty:
            return
        y = hit.iloc[0].get("Yield")
        chg = hit.iloc[0].get("Δ(bp)")
        rows.append({"Asset": label, "Price": y, "Chg (bp/%)": f"{chg} bp" if chg not in (None, "", "N/A") else ""})

    add_yield_row("2Y", "US 2Y Yield")
    add_yield_row("5Y", "US 5Y Yield")
    add_yield_row("10Y", "US 10Y Yield")
    add_yield_row("30Y", "US 30Y Yield")

    # key spreads
    if ust_hist is not None and not ust_hist.empty:
        sp = compute_key_spreads_bp(ust_hist)
        if "2s10s" in sp and not is_nan(sp["2s10s"].get("level_bp")):
            rows.append({
                "Asset": "UST 2s10s",
                "Price": fmt_signed(sp["2s10s"]["level_bp"], 1) + " bp",
                "Chg (bp/%)": fmt_signed(sp["2s10s"]["chg_bp"], 1) + " bp" if not is_nan(sp["2s10s"]["chg_bp"]) else "",
            })
        if "5s30s" in sp and not is_nan(sp["5s30s"].get("level_bp")):
            rows.append({
                "Asset": "UST 5s30s",
                "Price": fmt_signed(sp["5s30s"]["level_bp"], 1) + " bp",
                "Chg (bp/%)": fmt_signed(sp["5s30s"]["chg_bp"], 1) + " bp" if not is_nan(sp["5s30s"]["chg_bp"]) else "",
            })

    # commodities: include Gold/Oil if present
    if comm_df is not None and not comm_df.empty:
        asset_col = None
        for col in ("Asset", "Name"):
            if col in comm_df.columns:
                asset_col = col
                break
        if asset_col:
            def get_comm(names: List[str]) -> Optional[pd.Series]:
                for nm in names:
                    hit = comm_df[comm_df[asset_col].astype(str) == nm]
                    if not hit.empty:
                        return hit.iloc[0]
                return None

            gold = get_comm(["Gold"])
            if gold is not None:
                price = gold.get("LastRaw", gold.get("Last"))
                chg_pct = gold.get("ChgPctRaw", gold.get("ChgPct"))
                rows.append({"Asset": "Gold", "Price": fmt_num(price, 3), "Chg (bp/%)": fmt_signed(chg_pct, 2, "%")})

            oil = get_comm(["Oil", "WTI Crude", "Brent"])
            if oil is not None:
                price = oil.get("LastRaw", oil.get("Last"))
                chg_pct = oil.get("ChgPctRaw", oil.get("ChgPct"))
                rows.append({"Asset": "Oil", "Price": fmt_num(price, 3), "Chg (bp/%)": fmt_signed(chg_pct, 2, "%")})

    return pd.DataFrame(rows)


def fetch_commodities_realtime() -> pd.DataFrame:
    """
    Global commodities (foreign futures realtime) via Sina:
      1) futures_hq_subscribe_exchange_symbol -> (symbol, code)
      2) futures_foreign_commodity_realtime(symbol="CL,OIL,GC,HG,NG,...")

    We pull a small watchlist: WTI, Brent, Gold, Copper, NatGas.
    """
    import akshare as ak

    watch = [
        ("NYMEX原油", "WTI Crude"),
        ("布伦特原油", "Brent"),
        ("COMEX黄金", "Gold"),
        ("COMEX铜", "Copper"),
        ("NYMEX天然气", "Nat Gas"),
    ]

    codes = []
    try:
        m = ak.futures_hq_subscribe_exchange_symbol()
        m = m.copy()
        m["symbol"] = m["symbol"].astype(str)
        for key, _ in watch:
            hit = m[m["symbol"].str.contains(key, na=False)]
            if not hit.empty:
                codes.append(str(hit.iloc[0]["code"]))
    except Exception:
        # fallback to known common codes if mapping fails
        codes = ["CL", "OIL", "GC", "HG", "NG"]

    codes = [c for c in codes if c and c != "nan"]
    if not codes:
        return pd.DataFrame()

    rt = ak.futures_foreign_commodity_realtime(symbol=",".join(codes))
    if rt is None or rt.empty:
        return pd.DataFrame()

    rt = rt.copy()

    # Normalize output columns (per AKShare doc)
    # columns include: 名称 最新价 涨跌额 涨跌幅 行情时间 日期 ...
    rename = {
        "名称": "NameCN",
        "最新价": "Last",
        "涨跌额": "Chg",
        "涨跌幅": "ChgPct",
        "行情时间": "Time",
        "日期": "Date",
    }
    for k, v in rename.items():
        if k in rt.columns:
            rt.rename(columns={k: v}, inplace=True)

    # Map to English labels where possible
    cn_to_en = {cn: en for cn, en in watch}
    if "NameCN" in rt.columns:
        rt["Asset"] = rt["NameCN"].map(lambda x: cn_to_en.get(str(x), str(x)))
    else:
        rt["Asset"] = rt.iloc[:, 0].astype(str)

    # numeric
    for c in ["Last", "Chg", "ChgPct"]:
        if c in rt.columns:
            rt[c] = pd.to_numeric(rt[c], errors="coerce")

    out = rt[["Asset", "Last", "Chg", "ChgPct", "Time", "Date"]].copy()
    out["LastRaw"] = out["Last"]
    out["ChgRaw"] = out["Chg"]
    out["ChgPctRaw"] = out["ChgPct"]
    out["Last"] = out["Last"].map(lambda x: fmt_num(x, 3))
    out["Chg"] = out["Chg"].map(lambda x: fmt_signed(x, 3))
    out["ChgPct"] = out["ChgPct"].map(lambda x: fmt_signed(x, 2, "%"))
    return out


def plot_commodities_bar(comm_df: pd.DataFrame) -> Optional[str]:
    if comm_df is None or comm_df.empty:
        return None
    tmp = comm_df.copy()
    if "ChgPctRaw" in tmp.columns:
        tmp["ChgPctF"] = pd.to_numeric(tmp["ChgPctRaw"], errors="coerce")
    else:
        def pct_to_float(s: str) -> float:
            try:
                return float(str(s).replace("%", ""))
            except Exception:
                return float("nan")
        tmp["ChgPctF"] = tmp["ChgPct"].map(pct_to_float)

    tmp = tmp.dropna(subset=["ChgPctF"])
    if tmp.empty:
        return None

    fig = plt.figure(figsize=(10.5, 3.6))
    ax = fig.add_subplot(111)
    bars = ax.bar(tmp["Asset"].astype(str), tmp["ChgPctF"].astype(float))
    title = "Commodities % Move (Realtime)"
    if "Date" in tmp.columns and "Time" in tmp.columns:
        date_val = tmp["Date"].dropna().astype(str)
        time_val = tmp["Time"].dropna().astype(str)
        if not date_val.empty and not time_val.empty:
            title = f"Commodities % Move (Realtime @ {date_val.iloc[0]} {time_val.max()})"
    ax.set_title(title)
    ax.set_ylabel("Change (%)")
    ax.grid(True, axis="y", alpha=0.2)
    ax.tick_params(axis="x", rotation=20)
    for rect, val in zip(bars, tmp["ChgPctF"].astype(float)):
        if is_nan(val):
            continue
        label = f"{val:+.2f}%"
        height = rect.get_height()
        va = "bottom" if height >= 0 else "top"
        offset = 3 if height >= 0 else -3
        ax.annotate(
            label,
            (rect.get_x() + rect.get_width() / 2, height),
            xytext=(0, offset),
            textcoords="offset points",
            ha="center",
            va=va,
            fontsize=8,
        )
    return fig_to_base64(fig)


def fetch_fx_dashboard() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, str]]:
    """
    FX dashboard:
      - G10: EURUSD, USDJPY, GBPUSD from forex_spot_em
      - RMB: USD/CNY (onshore spot mid), USD/CNH (offshore), PBOC fixing
    """
    import akshare as ak

    def fmt_fx(x) -> str:
        if is_nan(x):
            return "N/A"
        try:
            val = float(x)
        except Exception:
            return "N/A"
        if val >= 100:
            nd = 2
        elif val >= 10:
            nd = 3
        else:
            nd = 4
        return fmt_num(val, nd)

    meta: Dict[str, str] = {}

    g10_rows = []
    rmb_rows = []
    swap_rows = []
    swap_df = pd.DataFrame()

    def try_fetch(fn, attempts: int = 2):
        for _ in range(attempts):
            try:
                df = fn()
                if df is not None and not df.empty:
                    return df
            except Exception:
                continue
        return None

    fx_em = try_fetch(ak.forex_spot_em, attempts=2)

    if fx_em is not None and not fx_em.empty:
        fx_em = fx_em.copy()
        fx_em["代码"] = fx_em["代码"].astype(str)
        for c in ["最新价", "涨跌额", "涨跌幅", "昨收"]:
            if c in fx_em.columns:
                fx_em[c] = pd.to_numeric(fx_em[c], errors="coerce")

        g10_codes = ["EURUSD", "USDJPY", "GBPUSD"]
        g10 = fx_em[fx_em["代码"].isin(g10_codes)]
        for _, r in g10.iterrows():
            last = r.get("最新价")
            prev = r.get("昨收")
            chg = r.get("涨跌额")
            chg_pct = r.get("涨跌幅")
            if is_nan(chg) and not is_nan(last) and not is_nan(prev):
                chg = last - prev
            if is_nan(chg_pct) and not is_nan(chg) and not is_nan(prev) and prev != 0:
                chg_pct = chg / prev * 100.0
            g10_rows.append({
                "Pair": r.get("代码"),
                "Last": fmt_fx(last),
                "Chg": fmt_signed(chg, 4),
                "Chg%": fmt_signed(chg_pct, 2, "%"),
                "Prev": fmt_fx(prev),
            })
    if not g10_rows:
        hist_map = [("EURUSD", "EURUSD"), ("GBPUSD", "GBPUSD"), ("USDJPY", "USDJPY")]
        for code, label in hist_map:
            hist = try_fetch(lambda symbol=code: ak.forex_hist_em(symbol=symbol), attempts=2)
            if hist is None or hist.empty:
                continue
            hist = hist.copy()
            date_col = "日期" if "日期" in hist.columns else ("date" if "date" in hist.columns else None)
            if date_col:
                hist[date_col] = pd.to_datetime(hist[date_col], errors="coerce")
                hist = hist.dropna(subset=[date_col]).sort_values(date_col)
            price_col = None
            for pc in ["最新价", "收盘价", "close", "Close", "last", "Last"]:
                if pc in hist.columns:
                    price_col = pc
                    break
            if not price_col or hist.empty:
                continue
            s = pd.to_numeric(hist[price_col], errors="coerce").dropna()
            if s.empty:
                continue
            last = float(s.iloc[-1])
            prev = float(s.iloc[-2]) if len(s) >= 2 else float("nan")
            chg = last - prev if not is_nan(prev) else float("nan")
            chg_pct = (chg / prev * 100.0) if not is_nan(prev) and prev != 0 else float("nan")
            g10_rows.append({
                "Pair": label,
                "Last": fmt_fx(last),
                "Chg": fmt_signed(chg, 4),
                "Chg%": fmt_signed(chg_pct, 2, "%"),
                "Prev": fmt_fx(prev),
            })
    if not g10_rows:
        try:
            pair = try_fetch(ak.fx_pair_quote, attempts=2)
            if pair is not None and not pair.empty:
                pair = pair.copy()
                pair["货币对"] = pair["货币对"].astype(str)
                for code in ["EUR/USD", "GBP/USD", "USD/JPY"]:
                    row = pair[pair["货币对"] == code]
                    if row.empty:
                        continue
                    r = row.iloc[0]
                    bid = pd.to_numeric(r.get("买报价"), errors="coerce")
                    ask = pd.to_numeric(r.get("卖报价"), errors="coerce")
                    mid = (float(bid) + float(ask)) / 2.0 if not is_nan(bid) and not is_nan(ask) else float("nan")
                    g10_rows.append({
                        "Pair": code.replace("/", ""),
                        "Last": fmt_fx(mid),
                        "Chg": "N/A",
                        "Chg%": "N/A",
                        "Prev": "N/A",
                    })
        except Exception:
            pass

    if g10_rows:
        order = {"EURUSD": 0, "GBPUSD": 1, "USDJPY": 2}
        g10_rows.sort(key=lambda x: order.get(str(x.get("Pair")), 99))

    g10_df = pd.DataFrame(g10_rows)

    # USD/CNY onshore spot mid (from ChinaMoney spot quote)
    cny_mid = float("nan")
    cny_prev = float("nan")
    cny_chg = float("nan")
    cny_chg_pct = float("nan")
    onshore_note = "N/A"
    try:
        spot = try_fetch(ak.fx_spot_quote, attempts=2)
        if spot is not None and not spot.empty:
            row = spot[spot["货币对"] == "USD/CNY"].iloc[0]
            bid = pd.to_numeric(row.get("买报价", row.get("买价")), errors="coerce")
            ask = pd.to_numeric(row.get("卖报价", row.get("卖价")), errors="coerce")
            if not is_nan(bid) and not is_nan(ask):
                cny_mid = (float(bid) + float(ask)) / 2.0
                onshore_note = f"Bid {fmt_fx(bid)} / Ask {fmt_fx(ask)}"
    except Exception:
        pass

    # USD/CNH offshore
    cnh_last = float("nan")
    cnh_prev = float("nan")
    if fx_em is not None and not fx_em.empty:
        cnh_row = fx_em[fx_em["代码"] == "USDCNH"]
        if not cnh_row.empty:
            r = cnh_row.iloc[0]
            cnh_last = r.get("最新价")
            prev = r.get("昨收")
            cnh_prev = prev
            chg = r.get("涨跌额")
            chg_pct = r.get("涨跌幅")
            if is_nan(chg) and not is_nan(cnh_last) and not is_nan(prev):
                chg = cnh_last - prev
            if is_nan(chg_pct) and not is_nan(chg) and not is_nan(prev) and prev != 0:
                chg_pct = chg / prev * 100.0
            rmb_rows.append({
                "Item": "USD/CNH (Offshore)",
                "Last": fmt_fx(cnh_last),
                "Prev": fmt_fx(prev),
                "Chg": fmt_signed(chg, 4),
                "Chg%": fmt_signed(chg_pct, 2, "%"),
                "Note": "Offshore spot",
            })
    if is_nan(cnh_last):
        try:
            hist = try_fetch(lambda: ak.forex_hist_em(symbol="USDCNH"), attempts=2)
            if hist is not None and not hist.empty:
                hist = hist.copy()
                hist["日期"] = pd.to_datetime(hist["日期"], errors="coerce")
                hist = hist.dropna(subset=["日期"]).sort_values("日期")
                last = hist.iloc[-1]
                prev = hist.iloc[-2] if len(hist) >= 2 else last
                cnh_last = pd.to_numeric(last.get("最新价"), errors="coerce")
                prev_val = pd.to_numeric(prev.get("最新价"), errors="coerce")
                cnh_prev = prev_val
                chg = cnh_last - prev_val if not is_nan(cnh_last) and not is_nan(prev_val) else float("nan")
                chg_pct = chg / prev_val * 100.0 if not is_nan(chg) and not is_nan(prev_val) and prev_val != 0 else float("nan")
                rmb_rows.append({
                    "Item": "USD/CNH (Offshore)",
                    "Last": fmt_fx(cnh_last),
                    "Prev": fmt_fx(prev_val),
                    "Chg": fmt_signed(chg, 4),
                    "Chg%": fmt_signed(chg_pct, 2, "%"),
                    "Note": "Offshore spot (hist)",
                })
        except Exception:
            pass

    # PBOC fixing (USD/CNYC)
    fixing = float("nan")
    fix_prev = float("nan")
    if fx_em is not None and not fx_em.empty:
        fix_row = fx_em[fx_em["代码"] == "USDCNYC"]
        if not fix_row.empty:
            r = fix_row.iloc[0]
            last = r.get("最新价")
            fixing = last
            prev = r.get("昨收")
            fix_prev = prev
            chg = r.get("涨跌额")
            chg_pct = r.get("涨跌幅")
            if is_nan(chg) and not is_nan(last) and not is_nan(prev):
                chg = last - prev
            if is_nan(chg_pct) and not is_nan(chg) and not is_nan(prev) and prev != 0:
                chg_pct = chg / prev * 100.0
            rmb_rows.append({
                "Item": "Fixing (PBOC Mid)",
                "Last": fmt_fx(last),
                "Prev": fmt_fx(prev),
                "Chg": fmt_signed(chg, 4),
                "Chg%": fmt_signed(chg_pct, 2, "%"),
                "Note": "USDCNYC",
            })
    if is_nan(fixing):
        try:
            hist_fix = try_fetch(lambda: ak.forex_hist_em(symbol="USDCNYC"), attempts=2)
            if hist_fix is not None and not hist_fix.empty:
                hist_fix = hist_fix.copy()
                hist_fix["日期"] = pd.to_datetime(hist_fix["日期"], errors="coerce")
                hist_fix = hist_fix.dropna(subset=["日期"]).sort_values("日期")
                last = hist_fix.iloc[-1]
                prev = hist_fix.iloc[-2] if len(hist_fix) >= 2 else last
                fixing = pd.to_numeric(last.get("最新价"), errors="coerce")
                prev_val = pd.to_numeric(prev.get("最新价"), errors="coerce")
                fix_prev = prev_val
                chg = fixing - prev_val if not is_nan(fixing) and not is_nan(prev_val) else float("nan")
                chg_pct = chg / prev_val * 100.0 if not is_nan(chg) and not is_nan(prev_val) and prev_val != 0 else float("nan")
                rmb_rows.append({
                    "Item": "Fixing (PBOC Mid)",
                    "Last": fmt_fx(fixing),
                    "Prev": fmt_fx(prev_val),
                    "Chg": fmt_signed(chg, 4),
                    "Chg%": fmt_signed(chg_pct, 2, "%"),
                    "Note": "USDCNYC (hist)",
                })
        except Exception:
            pass

    # try deriving previous close from CNYUSD (Baidu) if needed
    try:
        fx_bd = try_fetch(ak.fx_quote_baidu, attempts=2)
        if fx_bd is not None and not fx_bd.empty:
            row = fx_bd[fx_bd["代码"] == "CNYUSD"]
            if not row.empty:
                last = pd.to_numeric(row.iloc[0].get("最新价"), errors="coerce")
                chg = pd.to_numeric(row.iloc[0].get("涨跌额"), errors="coerce")
                if not is_nan(last) and last != 0:
                    if is_nan(cny_mid):
                        cny_mid = 1.0 / float(last)
                        onshore_note = "Derived from CNYUSD (Baidu)"
                    if not is_nan(chg):
                        prev_cnyusd = last - chg
                        if not is_nan(prev_cnyusd) and prev_cnyusd != 0:
                            cny_prev = 1.0 / float(prev_cnyusd)
                            if onshore_note != "N/A":
                                onshore_note = f"{onshore_note} | Prev from CNYUSD (Baidu)"
                            else:
                                onshore_note = "Prev from CNYUSD (Baidu)"
    except Exception:
        pass

    if is_nan(cny_mid) and not is_nan(fixing):
        cny_mid = float(fixing)
        onshore_note = "Proxy from fixing"
    if is_nan(cny_mid) and not is_nan(cnh_last):
        cny_mid = float(cnh_last)
        onshore_note = "Proxy from CNH"

    if is_nan(cny_prev) and not is_nan(cnh_prev):
        cny_prev = float(cnh_prev)
        onshore_note = f"{onshore_note} | Prev proxy: CNH" if onshore_note != "N/A" else "Prev proxy: CNH"
    if is_nan(cny_prev) and not is_nan(fix_prev):
        cny_prev = float(fix_prev)
        onshore_note = f"{onshore_note} | Prev proxy: fixing" if onshore_note != "N/A" else "Prev proxy: fixing"

    if (not is_nan(cny_mid)) and (not is_nan(cny_prev)) and cny_prev:
        cny_chg = cny_mid - cny_prev
        cny_chg_pct = cny_chg / cny_prev * 100.0

    rmb_rows.insert(0, {
        "Item": "USD/CNY (Onshore Spot)",
        "Last": fmt_fx(cny_mid) if not is_nan(cny_mid) else "N/A",
        "Prev": fmt_fx(cny_prev) if not is_nan(cny_prev) else "N/A",
        "Chg": fmt_signed(cny_chg, 4) if not is_nan(cny_chg) else "N/A",
        "Chg%": fmt_signed(cny_chg_pct, 2, "%") if not is_nan(cny_chg_pct) else "N/A",
        "Note": onshore_note if not is_nan(cny_mid) else "N/A",
    })

    # Spot-Fixing spread (CNY spot - fixing)
    spot_fix_spread = float("nan")
    spot_fix_pips = float("nan")
    spot_fix_prev = float("nan")
    spot_fix_chg_pips = float("nan")
    if not is_nan(cny_mid) and not is_nan(fixing):
        spot_fix_spread = float(cny_mid) - float(fixing)
        spot_fix_pips = spot_fix_spread * 10000.0
    if not is_nan(cny_prev) and not is_nan(fix_prev):
        spot_fix_prev = float(cny_prev) - float(fix_prev)
        spot_fix_chg_pips = (spot_fix_spread - spot_fix_prev) * 10000.0 if not is_nan(spot_fix_spread) else float("nan")
    spot_fix_note = "N/A"
    if not is_nan(spot_fix_pips):
        if abs(spot_fix_pips) >= 200:
            spot_fix_note = "Wide: spot above fix" if spot_fix_pips > 0 else "Wide: spot below fix"
        elif abs(spot_fix_pips) >= 100:
            spot_fix_note = "Moderate"
        else:
            spot_fix_note = "Neutral"
    rmb_rows.append({
        "Item": "Spot-Fixing Spread",
        "Last": fmt_fx(spot_fix_spread) if not is_nan(spot_fix_spread) else "N/A",
        "Prev": fmt_fx(spot_fix_prev) if not is_nan(spot_fix_prev) else "N/A",
        "Chg": (
            fmt_signed(spot_fix_chg_pips, 0, " pips")
            if not is_nan(spot_fix_chg_pips)
            else (fmt_signed(spot_fix_pips, 0, " pips") if not is_nan(spot_fix_pips) else "N/A")
        ),
        "Chg%": "",
        "Note": spot_fix_note,
    })

    # CNH-CNY spread
    spread = float("nan")
    spread_pips = float("nan")
    spread_prev = float("nan")
    spread_chg_pips = float("nan")
    if not is_nan(cnh_last) and not is_nan(cny_mid):
        spread = float(cnh_last) - float(cny_mid)
        spread_pips = spread * 10000.0
    if not is_nan(cnh_prev) and not is_nan(cny_prev):
        spread_prev = float(cnh_prev) - float(cny_prev)
        spread_chg_pips = (spread - spread_prev) * 10000.0 if not is_nan(spread) else float("nan")
    spread_note = "N/A"
    if not is_nan(spread_pips):
        if abs(spread_pips) >= 200:
            spread_note = "Wide: CNH weaker" if spread_pips > 0 else "Wide: CNH stronger"
        elif abs(spread_pips) >= 100:
            spread_note = "Moderate spread"
        else:
            spread_note = "Neutral"
    rmb_rows.append({
        "Item": "CNH - CNY Spread",
        "Last": fmt_fx(spread) if not is_nan(spread) else "N/A",
        "Prev": fmt_fx(spread_prev) if not is_nan(spread_prev) else "N/A",
        "Chg": (
            fmt_signed(spread_chg_pips, 0, " pips")
            if not is_nan(spread_chg_pips)
            else (fmt_signed(spread_pips, 0, " pips") if not is_nan(spread_pips) else "N/A")
        ),
        "Chg%": "",
        "Note": spread_note,
    })

    meta["spread_note"] = spread_note
    rmb_df = pd.DataFrame(rmb_rows)

    # Swap points (USD/CNY)
    try:
        swap = try_fetch(ak.fx_swap_quote, attempts=2)
        if swap is not None and not swap.empty:
            swap["货币对"] = swap["货币对"].astype(str)
            row = swap[swap["货币对"].str.contains("USD/CNY", na=False)]
            if not row.empty:
                r = row.iloc[0]
                tenor_map = {"1周": "1W", "1月": "1M", "3月": "3M", "6月": "6M", "9月": "9M", "1年": "1Y"}
                for tenor in ["1周", "1月", "3月", "6月", "9月", "1年"]:
                    val = str(r.get(tenor, "")).strip()
                    if "/" in val:
                        bid_str, ask_str = val.split("/", 1)
                    else:
                        bid_str, ask_str = val, ""
                    bid = pd.to_numeric(bid_str, errors="coerce")
                    ask = pd.to_numeric(ask_str, errors="coerce")
                    mid = (float(bid) + float(ask)) / 2.0 if not is_nan(bid) and not is_nan(ask) else float("nan")
                    swap_rows.append({
                        "Tenor": tenor_map.get(tenor, tenor),
                        "Bid": fmt_num(bid, 2) if not is_nan(bid) else "N/A",
                        "Ask": fmt_num(ask, 2) if not is_nan(ask) else "N/A",
                        "Mid": fmt_num(mid, 2) if not is_nan(mid) else "N/A",
                    })
    except Exception:
        pass
    swap_df = pd.DataFrame(swap_rows)
    if swap_df is not None and swap_df.empty:
        swap_df = pd.DataFrame([{
            "Tenor": "N/A",
            "Bid": "N/A",
            "Ask": "N/A",
            "Mid": "N/A",
        }])
    return g10_df, rmb_df, swap_df, meta


def fetch_china_pmi_suite() -> Tuple[pd.DataFrame, Dict[str, pd.Series]]:
    """
    China PMI prints (Jin10 via AKShare):
      - macro_china_pmi_yearly: Official Manufacturing PMI
      - macro_china_non_man_pmi: Official Non-manufacturing PMI
      - macro_china_cx_pmi_yearly: Caixin Manufacturing PMI Final
      - macro_china_cx_services_pmi_yearly: Caixin Services PMI

    Returns:
      snapshot table + series dict for charting (last ~36 points)
    """
    import akshare as ak

    def load_cx_index_series(fn_name: str, value_col: str) -> Optional[pd.Series]:
        try:
            fn = getattr(ak, fn_name)
            df_idx = fn()
        except Exception:
            return None
        if df_idx is None or df_idx.empty:
            return None
        if "日期" not in df_idx.columns or value_col not in df_idx.columns:
            return None
        df_idx = df_idx.copy()
        df_idx["日期"] = pd.to_datetime(df_idx["日期"], errors="coerce")
        df_idx[value_col] = pd.to_numeric(df_idx[value_col], errors="coerce")
        df_idx = df_idx.dropna(subset=["日期", value_col]).sort_values("日期")
        if df_idx.empty:
            return None
        return df_idx.set_index("日期")[value_col]

    def merge_series(primary: Optional[pd.Series], fallback: Optional[pd.Series]) -> Optional[pd.Series]:
        if primary is None or primary.empty:
            return fallback
        if fallback is None or fallback.empty:
            return primary
        cutoff = primary.index.max()
        extra = fallback[fallback.index > cutoff]
        if extra.empty:
            return primary
        return pd.concat([primary, extra]).sort_index()

    suite = [
        ("Official Mfg PMI", "macro_china_pmi_yearly"),
        ("Official Non-mfg PMI", "macro_china_non_man_pmi"),
        ("Caixin Mfg PMI", "macro_china_cx_pmi_yearly"),
        ("Caixin Services PMI", "macro_china_cx_services_pmi_yearly"),
    ]

    rows = []
    series_map: Dict[str, pd.Series] = {}

    for label, fn_name in suite:
        try:
            fn = getattr(ak, fn_name)
            df = fn()
            if df is None or df.empty:
                continue
            df = df.copy()
            # expected cols: 商品 日期 今值 预测值 前值
            if "日期" in df.columns:
                df["日期"] = pd.to_datetime(df["日期"], errors="coerce")
                df = df.dropna(subset=["日期"]).sort_values("日期", ascending=False)

            for c in ["今值", "预测值", "前值"]:
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors="coerce")

            valid_actual = df.dropna(subset=["今值"]) if "今值" in df.columns else df
            last = valid_actual.iloc[0] if not valid_actual.empty else df.iloc[0]

            actual = last.get("今值")
            fcst = last.get("预测值")
            prev = last.get("前值")
            surprise = actual - fcst if (not is_nan(actual) and not is_nan(fcst)) else float("nan")

            rows.append({
                "Indicator": label,
                "Date": (pd.to_datetime(last.get("日期")).date().isoformat()
                         if not is_nan(last.get("日期")) else "N/A"),
                "Actual": fmt_num(actual, 1),
                "Forecast": fmt_num(fcst, 1),
                "Previous": fmt_num(prev, 1),
                "Surprise": fmt_signed(surprise, 1),
            })

            # series (last ~40)
            s = df[["日期", "今值"]].dropna()
            s["今值"] = pd.to_numeric(s["今值"], errors="coerce")
            s = s.dropna().sort_values("日期")
            if not s.empty:
                series_map[label] = s.set_index("日期")["今值"].tail(48)
        except Exception:
            rows.append({
                "Indicator": label,
                "Date": "N/A",
                "Actual": "N/A",
                "Forecast": "N/A",
                "Previous": "N/A",
                "Surprise": "N/A",
            })

    snapshot = pd.DataFrame(rows)
    # Caixin indices can be more up-to-date; merge for chart continuity.
    cx_mfg = merge_series(series_map.get("Caixin Mfg PMI"), load_cx_index_series("index_pmi_man_cx", "制造业PMI"))
    if cx_mfg is not None and not cx_mfg.empty:
        series_map["Caixin Mfg PMI"] = cx_mfg.tail(48)
    cx_srv = merge_series(series_map.get("Caixin Services PMI"), load_cx_index_series("index_pmi_ser_cx", "服务业PMI"))
    if cx_srv is not None and not cx_srv.empty:
        series_map["Caixin Services PMI"] = cx_srv.tail(48)
    return snapshot, series_map


def plot_pmi(series_map: Dict[str, pd.Series]) -> Optional[str]:
    if not series_map:
        return None

    fig = plt.figure(figsize=(10.5, 4.0))
    ax = fig.add_subplot(111)
    for label, series in series_map.items():
        s = series.dropna().sort_index()
        if s.empty:
            continue
        ax.plot(s.index, s.values, label=label)
    ax.axhline(50, linestyle="--", linewidth=1)
    ax.set_title("China PMI (Latest ~48 prints)")
    ax.set_ylabel("PMI")
    ax.grid(True, alpha=0.2)
    ax.legend(loc="best", fontsize=8)
    return fig_to_base64(fig)


# ----------------------------
# Cross-Asset Correlations
# ----------------------------

def fetch_dxy_series(days_back: int = 260) -> pd.Series:
    import akshare as ak

    # NOTE:
    # - `ak.index_us_stock_sina` only officially supports a small set of US indices.
    #   Passing "DX" can silently return the *US stock ticker* DX (Dynex Capital),
    #   which is NOT the Dollar Index (DXY).
    # - For DXY, prefer the global index history APIs.

    # 1) Eastmoney global index history (recommended)
    try:
        df = ak.index_global_hist_em(symbol="美元指数")
        if df is not None and not df.empty:
            df = df.copy()
            date_col = "日期" if "日期" in df.columns else ("date" if "date" in df.columns else None)
            price_col = None
            for c in ["最新价", "收盘", "close", "Close"]:
                if c in df.columns:
                    price_col = c
                    break
            if date_col and price_col:
                df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
                df = df.dropna(subset=[date_col]).sort_values(date_col)
                s = pd.to_numeric(df[price_col], errors="coerce")
                s = pd.Series(s.values, index=df[date_col]).dropna()
                # Sanity check: DXY level is typically ~70-130 (loose range).
                if not s.empty:
                    med = float(s.median())
                    if 50.0 <= med <= 200.0:
                        return s.tail(days_back)
    except Exception:
        pass

    # 2) Sina global index history (fallback)
    try:
        df = ak.index_global_hist_sina(symbol="美元指数")
        if df is not None and not df.empty and "date" in df.columns and "close" in df.columns:
            df = df.copy()
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.dropna(subset=["date"]).sort_values("date")
            s = pd.to_numeric(df["close"], errors="coerce")
            s = pd.Series(s.values, index=df["date"]).dropna()
            if not s.empty:
                med = float(s.median())
                if 50.0 <= med <= 200.0:
                    return s.tail(days_back)
    except Exception:
        pass

    # If we can't get a sensible DXY series, return empty (better than wrong data).
    return pd.Series(dtype=float)


def fetch_gold_series(days_back: int = 260) -> pd.Series:
    import akshare as ak

    df = ak.futures_foreign_hist(symbol="GC")
    if df is None or df.empty:
        return pd.Series(dtype=float)
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date")
    s = pd.to_numeric(df["close"], errors="coerce")
    return pd.Series(s.values, index=df["date"]).tail(days_back)


def build_cross_asset_corr(window: int = 20) -> Tuple[pd.DataFrame, str]:
    dxy = fetch_dxy_series()
    gold = fetch_gold_series()

    df = pd.DataFrame({"DXY": dxy, "Gold": gold}).dropna().sort_index()
    if df.empty or len(df) < window + 2:
        return pd.DataFrame(), "Insufficient data for rolling correlation."

    df["DXY_ret"] = df["DXY"].pct_change()
    df["Gold_ret"] = df["Gold"].pct_change()
    df = df.dropna()
    df["RollingCorr"] = df["DXY_ret"].rolling(window).corr(df["Gold_ret"])

    last = df.iloc[-1]
    alert = ""
    if last["DXY_ret"] > 0 and last["Gold_ret"] > 0 and last["RollingCorr"] < 0:
        alert = "Divergence Alert: Gold resilience despite higher DXY."
    elif last["DXY_ret"] < 0 and last["Gold_ret"] < 0 and last["RollingCorr"] < 0:
        alert = "Divergence Alert: Gold sliding despite softer DXY."

    note = alert if alert else "Correlation within normal range."
    return df, note


def plot_rolling_corr(df: pd.DataFrame) -> Optional[str]:
    if df is None or df.empty or "RollingCorr" not in df.columns:
        return None
    fig = plt.figure(figsize=(10.5, 3.6))
    ax = fig.add_subplot(111)
    ax.plot(df.index, df["RollingCorr"], color="#f59e0b", linewidth=1.4)
    ax.axhline(0, color="#888", linewidth=1, linestyle="--")
    ax.set_title("Rolling Correlation: DXY vs Gold (20D)")
    ax.set_ylabel("Corr")
    ax.grid(True, alpha=0.2)
    return fig_to_base64(fig)


# ----------------------------
# Gemini Daily Color (optional)
# ----------------------------

def generate_daily_color(snapshot_text: str) -> str:
    """
    Uses google-genai SDK.
    If GEMINI_API_KEY is missing or SDK not installed, returns a fallback text.
    """
    model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip()
    api_key = os.getenv("GEMINI_API_KEY", "").strip()

    if not api_key:
        return (
            "GEMINI_API_KEY not configured. Charts and tables are generated, but Daily Color is skipped. "
            "Add GEMINI_API_KEY in .env to enable auto-generation."
        )

    try:
        from google import genai  # google-genai
        # Client reads GEMINI_API_KEY from env, but we pass it explicitly for clarity.
        client = genai.Client(api_key=api_key)

        sys_style = (
            "你是一名资深（senior）多资产交易员/宏观交易员，写给机构内部的 Morning Note。"
            "口吻要像 trader：简洁、直接、有观点、有风险提示，适当夹杂英文交易术语（risk-on/off, "
            "rates, curve, carry, positioning, levels）。\n"
            "硬性要求：\n"
            "1) 只能基于我给的数字和事实，不要编造数据/事件。\n"
            "2) 不要写“作为AI…”。\n"
            "3) 输出结构：\n"
            "   - Headline（一句话）\n"
            "   - Drivers（三条 bullet）\n"
            "   - Daily Color（1-2段）\n"
            "   - Risk (一句：今天最大的风险点)\n"
            "输出使用英文。\n"
        )

        prompt = f"{sys_style}\n\n【市场快照】\n{snapshot_text}\n"

        model_candidates = [m for m in [
            model,
            "gemini-2.5-flash",
            "gemini-2.0-flash",
            "gemini-1.5-flash",
            "gemini-1.5-pro",
        ] if m]
        seen = set()
        model_candidates = [m for m in model_candidates if not (m in seen or seen.add(m))]

        last_err: Optional[Exception] = None
        for m in model_candidates:
            try:
                resp = client.models.generate_content(
                    model=m,
                    contents=prompt,
                )
                text = getattr(resp, "text", None)
                if text:
                    return text.strip()
                last_err = RuntimeError("Gemini returned empty response")
            except Exception as e:
                last_err = e
                msg = str(e)
                if "NOT_FOUND" in msg or "not found" in msg or "not supported" in msg:
                    continue
                break

        if last_err is None:
            return "Daily Color failed: empty response from Gemini."
        return f"Daily Color failed (Gemini error: {type(last_err).__name__}: {last_err})"

    except Exception as e:
        return f"Daily Color failed (Gemini error: {type(e).__name__}: {e})"


# ----------------------------
# HTML Rendering
# ----------------------------

def format_inline(text: str) -> str:
    safe = html_lib.escape(text)
    safe = re.sub(r"\*\*(.+?)\*\*", r'<strong class="dc-key">\1</strong>', safe)
    return safe


def split_paragraphs(lines: List[str]) -> List[str]:
    paragraphs = []
    buf: List[str] = []
    for line in lines:
        if line == "":
            if buf:
                paragraphs.append(" ".join(buf))
                buf = []
            continue
        buf.append(line)
    if buf:
        paragraphs.append(" ".join(buf))
    return paragraphs


def format_daily_color_html(daily_color: str) -> str:
    if not daily_color:
        return "<div class='muted'>No content.</div>"

    lines = daily_color.splitlines()
    sections = {"Headline": [], "Drivers": [], "Daily Color": [], "Risk": []}
    preface: List[str] = []
    current: Optional[str] = None
    label_map = {
        "headline": "Headline",
        "drivers": "Drivers",
        "daily color": "Daily Color",
        "risk": "Risk",
    }

    for raw in lines:
        line = raw.strip()
        if not line:
            if current == "Daily Color":
                sections[current].append("")
            continue
        m = re.match(r"^\*{0,2}\s*(Headline|Drivers|Daily Color|Risk)\s*[:：]\*{0,2}\s*(.*)$", line, re.IGNORECASE)
        if m:
            key = label_map[m.group(1).lower()]
            current = key
            content = m.group(2).strip()
            if content:
                sections[key].append(content)
            continue

        if current == "Drivers":
            line = re.sub(r"^[-*•]\s+", "", line)
            if line:
                sections[current].append(line)
            continue

        if current:
            sections[current].append(line)
        else:
            preface.append(line)

    if not any(sections.values()):
        return f"<div class='dc-plain'>{format_inline(' '.join(preface))}</div>"

    parts = ["<div class='dc'>"]
    if preface:
        parts.append(f"<div class='dc-meta'>{format_inline(' '.join(preface))}</div>")
    if sections["Headline"]:
        parts.append(
            "<div class='dc-row dc-headline'>"
            "<div class='dc-label'>Headline</div>"
            f"<div class='dc-text'>{format_inline(' '.join(sections['Headline']))}</div>"
            "</div>"
        )
    if sections["Drivers"]:
        items = "".join(f"<li>{format_inline(item)}</li>" for item in sections["Drivers"])
        parts.append(
            "<div class='dc-row dc-drivers'>"
            "<div class='dc-label'>Drivers</div>"
            f"<ul>{items}</ul>"
            "</div>"
        )
    if sections["Daily Color"]:
        paras = split_paragraphs(sections["Daily Color"])
        body_html = "".join(f"<p>{format_inline(p)}</p>" for p in paras) if paras else ""
        parts.append(
            "<div class='dc-row dc-body'>"
            "<div class='dc-label'>Daily Color</div>"
            f"<div class='dc-text'>{body_html}</div>"
            "</div>"
        )
    if sections["Risk"]:
        parts.append(
            "<div class='dc-row dc-risk'>"
            "<div class='dc-label'>Risk</div>"
            f"<div class='dc-text'>{format_inline(' '.join(sections['Risk']))}</div>"
            "</div>"
        )
    parts.append("</div>")
    return "".join(parts)


def fetch_macro_calendar_raw(date: Optional[str] = None) -> pd.DataFrame:
    from akshare.news import news_baidu

    date_str = date or datetime.now().strftime("%Y%m%d")
    try:
        df = news_baidu.news_economic_baidu(date=date_str)
    except Exception:
        return pd.DataFrame()
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()
    return df


def _impact_label(val) -> str:
    if val is None:
        return "Low"
    s = str(val).strip().lower()
    if s in ("3", "high", "h"):
        return "High"
    if s in ("2", "medium", "med", "m"):
        return "Medium"
    if s in ("1", "low", "l"):
        return "Low"
    return str(val)


def _impact_html(val: str) -> str:
    label = _impact_label(val)
    cls = "low"
    if label.lower().startswith("h"):
        cls = "high"
    elif label.lower().startswith("m"):
        cls = "medium"
    return f"<span class='impact {cls}'>{label}</span>"


def _scenario_from_event(event: str) -> str:
    e = event.lower()
    if any(k in e for k in ["cpi", "pce", "ppi", "通胀", "物价"]):
        return "Hot print => UST yields higher, USD stronger, gold softer; cool print => the opposite."
    if any(k in e for k in ["非农", "就业", "失业", "adp", "payroll"]):
        return "Strong labor => yields up, USD bid; weak print => yields down, gold supported."
    if any(k in e for k in ["lpr", "mlf", "rrr", "降准", "降息", "利率下调"]):
        return "Easing signal => CNH softer, yields lower; no cut => CNH steadier."
    if any(k in e for k in ["利率", "央行", "决议", "fomc", "boj", "pbo", "pbc"]):
        return "Hawkish tilt => USD firmer & yields higher; dovish tilt => USD softer, gold bid."
    if any(k in e for k in ["gdp", "pmi", "零售", "工业产出", "消费"]):
        return "Growth surprise up => USD firmer, yields higher; downside miss => the opposite."
    if any(k in e for k in ["人民币", "中间价", "汇率"]):
        return "Stronger fix => CNH firmer; weaker fix => CNH softer."
    return "Surprise vs consensus drives USD/UST knee-jerk; watch CNH and gold."


def _macro_calendar_fallback(events_df: pd.DataFrame, max_items: int = 5) -> Tuple[pd.DataFrame, str]:
    if events_df is None or events_df.empty:
        return pd.DataFrame(), "No macro events available."

    df = events_df.copy()
    for c in ["重要性"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    keywords = ["CPI", "PCE", "PPI", "FOMC", "NFP", "ADP", "GDP", "PMI",
                "利率", "央行", "非农", "失业", "通胀", "就业", "零售", "国债", "中间价", "人民币"]
    regions = ["美国", "中国", "日本", "欧元区", "英国", "US", "China", "Japan", "Eurozone", "UK"]
    df["is_key"] = df["事件"].astype(str).str.contains("|".join(keywords), na=False)
    df["is_region"] = df["地区"].astype(str).isin(regions)
    filt = df[df["is_key"] | df["is_region"]].copy()
    noise_terms = ["持仓", "库存", "仓单", "更新", "ETF", "etf", "SPDR", "iShares"]
    filt["is_noise"] = filt["事件"].astype(str).str.contains("|".join(noise_terms), na=False)
    filt_main = filt[~filt["is_noise"]]
    if not filt_main.empty:
        filt = filt_main
    if filt.empty:
        filt = df
    filt = filt.sort_values(["重要性", "时间"], ascending=[False, True]).head(max_items)

    rows = []
    for _, r in filt.iterrows():
        impact = _impact_label(r.get("重要性", "Low"))
        scenario = _scenario_from_event(str(r.get("事件", "")))
        region = translate_region(str(r.get("地区", "")))
        event = translate_event_text(str(r.get("事件", "")))
        rows.append({
            "Time": r.get("时间", ""),
            "Region": region,
            "Event": event,
            "Impact": _impact_html(impact),
            "Scenario": scenario,
        })
    return pd.DataFrame(rows), "Auto-filtered (rule-based)."


def generate_macro_calendar(events_df: pd.DataFrame, max_items: int = 5) -> Tuple[pd.DataFrame, str]:
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if events_df is None or events_df.empty:
        return pd.DataFrame(), "No macro events available."

    if not api_key:
        return _macro_calendar_fallback(events_df, max_items)

    model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip()
    try:
        from google import genai
        client = genai.Client(api_key=api_key)
        records = []
        for _, r in events_df.iterrows():
            records.append({
                "time": r.get("时间", ""),
                "region": r.get("地区", ""),
                "event": r.get("事件", ""),
                "actual": r.get("公布", ""),
                "forecast": r.get("预期", ""),
                "previous": r.get("前值", ""),
                "importance": r.get("重要性", ""),
            })
        payload = "\n".join([str(rec) for rec in records[:80]])
        prompt = (
            "你是宏观交易员，筛选今天的财经事件给 S&T 晨报。\n"
            "规则：只保留对 USD、CNH、JPY、Gold、US Rates 影响最大的 Top 3-5 条；"
            "给每条事件打 Impact（High/Medium/Low）。\n"
            "对于 High 事件，用一句话给出 S&T 视角的 scenario（例如高于预期如何影响美债/美元/黄金）。\n"
            "输出必须是 JSON 数组，每个元素字段：time, region, event, impact, scenario。\n"
            "输出使用英文。\n"
            "如果没有事件，输出空数组 []。\n\n"
            f"事件列表:\n{payload}\n"
        )
        resp = client.models.generate_content(model=model, contents=prompt)
        text = getattr(resp, "text", "") or ""
        import json
        data = None
        try:
            data = json.loads(text)
        except Exception:
            m = re.search(r"\\[.*\\]", text, re.S)
            if m:
                try:
                    data = json.loads(m.group(0))
                except Exception:
                    data = None
        if not isinstance(data, list) or not data:
            return _macro_calendar_fallback(events_df, max_items)
        rows = []
        for item in data[:max_items]:
            impact = _impact_label(item.get("impact"))
            scenario = item.get("scenario", "") or _scenario_from_event(str(item.get("event", "")))
            region = translate_region(str(item.get("region", "")))
            event = translate_event_text(str(item.get("event", "")))
            rows.append({
                "Time": item.get("time", ""),
                "Region": region,
                "Event": event,
                "Impact": _impact_html(impact),
                "Scenario": scenario,
            })
        return pd.DataFrame(rows), "Gemini filtered."
    except Exception:
        return _macro_calendar_fallback(events_df, max_items)


def render_macro_calendar_html(macro_df: pd.DataFrame, note: str) -> str:
    if macro_df is None or macro_df.empty:
        return f"<div class='muted'>{html_lib.escape(note)}</div>"
    table_html = df_to_html_table(macro_df, colorize_cols=[])
    note_html = f"<div class='muted'>{html_lib.escape(note)}</div>" if note else ""
    return f"{table_html}{note_html}"


def render_html(
    generated_at: str,
    daily_color: str,
    us_eq_tbl: str,
    us_eq_chart_b64: Optional[str],
    us_eq_chart_conclusion: str,
    us_market_note: str,
    fx_g10_tbl: str,
    fx_rmb_tbl: str,
    fx_spread_note: str,
    fx_swap_tbl: str,
    rates_comm_tbl: str,
    ust_tbl: str,
    ust_spreads_tbl: str,
    ust_section_conclusion: str,
    ust_curve_b64: Optional[str],
    ust_curve_conclusion: str,
    ust_curve_compare_b64: Optional[str],
    ust_curve_compare_conclusion: str,
    ust_2s10s_b64: Optional[str],
    ust_2s10s_conclusion: str,
    ust_5s30s_b64: Optional[str],
    ust_5s30s_conclusion: str,
    comm_tbl: str,
    comm_bar_b64: Optional[str],
    comm_bar_conclusion: str,
    pmi_tbl: str,
    pmi_chart_b64: Optional[str],
    pmi_chart_conclusion: str,
    corr_tbl: str,
    corr_chart_b64: Optional[str],
    corr_chart_conclusion: str,
    corr_note: str,
    macro_html: str,
    data_notes: str,
) -> str:
    def img_block(b64: Optional[str]) -> str:
        if not b64:
            return "<div class='muted'>Chart unavailable.</div>"
        return f"<img src='data:image/png;base64,{b64}' />"

    def chart_block(b64: Optional[str], conclusion: str = "") -> str:
        out = img_block(b64)
        if conclusion:
            out += f"<div class='conclusion'>{html_lib.escape(conclusion)}</div>"
        return out

    daily_color_html = format_daily_color_html(daily_color)

    foot_html = ""
    if data_notes:
        escaped_notes = html_lib.escape(data_notes).replace("\n", "<br>")
        foot_html = (
            "<div class=\"foot\">"
            "<div><b>Data Notes</b></div>"
            f"<div>{escaped_notes}</div>"
            "</div>"
        )

    spread_class = "callout alert" if "Wide" in fx_spread_note else "callout"
    market_class = "callout alert" if "Closed" in us_market_note else "callout"
    corr_class = "callout alert" if "Divergence Alert" in corr_note else "callout"
    market_note_html = f"<div class=\"{market_class}\">{html_lib.escape(us_market_note)}</div>" if us_market_note else ""

    return f"""<!doctype html>
<html lang=\"zh-CN\">
<head>
<meta charset=\"utf-8\" />
<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
<title>Morning Note</title>
<style>
  :root {{
    --bg: #0b0f17;
    --panel: #0f172a;
    --panel-2: #111827;
    --text: #e5e7eb;
    --muted: #94a3b8;
    --border: #1f2937;
    --pos: #22c55e;
    --neg: #ef4444;
    --accent: #38bdf8;
    --warn: #f59e0b;
    --risk: #f87171;
  }}
  body {{
    font-family: "Menlo", "SF Mono", Consolas, "Liberation Mono", "Courier New", monospace;
    margin: 24px;
    color: var(--text);
    background: var(--bg);
  }}
  h1 {{ margin: 0 0 4px 0; letter-spacing: 0.02em; }}
  h2 {{ margin: 0 0 8px 0; }}
  h3 {{ margin: 0 0 6px 0; font-size: 13px; color: var(--accent); }}
  .sub {{ color: var(--muted); font-size: 12px; margin-bottom: 18px; }}
  .section {{ margin-top: 22px; padding-top: 12px; border-top: 1px solid var(--border); }}
  .note {{
    white-space: normal;
    font-size: 13px;
    line-height: 1.55;
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 14px 16px;
  }}
  .grid-2 {{
    display: grid;
    gap: 12px;
    grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
    grid-auto-flow: dense;
    align-items: start;
  }}
  .grid-3 {{
    display: grid;
    gap: 10px;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    grid-auto-flow: dense;
    align-items: start;
  }}
  .card {{
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 10px 12px;
    align-self: start;
  }}
  .callout {{
    background: #0b1220;
    border: 1px solid #1d4ed8;
    color: #bfdbfe;
    padding: 8px 10px;
    border-radius: 10px;
    font-size: 12px;
  }}
  .callout.alert {{
    border-color: #ef4444;
    color: #fecaca;
    background: #1f0b0b;
  }}
  .conclusion {{
    margin-top: 8px;
    color: var(--muted);
    font-size: 12px;
    line-height: 1.45;
  }}
  h4 {{
    margin: 10px 0 6px 0;
    font-size: 12px;
    color: var(--muted);
    letter-spacing: 0.02em;
  }}
  .dc {{
    display: grid;
    gap: 10px;
  }}
  .dc-row {{
    padding-left: 10px;
    border-left: 4px solid var(--border);
  }}
  .dc-label {{
    display: inline-block;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 4px;
    color: var(--muted);
  }}
  .dc-text {{
    margin-top: 2px;
  }}
  .dc-text p {{ margin: 6px 0; }}
  .dc-meta {{
    color: var(--muted);
    font-size: 12px;
  }}
  .dc-plain {{
    color: var(--text);
  }}
  .dc-headline {{ border-color: #38bdf8; }}
  .dc-headline .dc-label {{ color: #7dd3fc; }}
  .dc-drivers {{ border-color: #f59e0b; }}
  .dc-drivers .dc-label {{ color: #fdba74; }}
  .dc-setup {{ border-color: #22c55e; }}
  .dc-setup .dc-label {{ color: #86efac; }}
  .dc-risks {{ border-color: #ef4444; }}
  .dc-risks .dc-label {{ color: #fecaca; }}
  .tbl {{
    width: 100%;
    border-collapse: collapse;
    font-size: 12px;
  }}
  .tbl th {{
    text-align: left;
    font-weight: 700;
    color: var(--muted);
    padding: 6px 8px;
    border-bottom: 1px solid var(--border);
  }}
  .tbl td {{
    padding: 6px 8px;
    border-bottom: 1px solid var(--border);
    vertical-align: top;
  }}
  .pos {{ color: var(--pos); font-weight: 700; }}
  .neg {{ color: var(--neg); font-weight: 700; }}
  .muted {{ color: var(--muted); }}
  img {{
    width: 100%;
    border-radius: 10px;
    border: 1px solid var(--border);
    background: var(--panel-2);
  }}
  .foot {{
    margin-top: 22px;
    color: var(--muted);
    font-size: 12px;
    border-top: 1px solid var(--border);
    padding-top: 10px;
  }}
</style>
</head>
<body>
  <h1>Morning Note</h1>
  <div class=\"sub\">Generated at: {generated_at}</div>

  <div class=\"section\">
    <h2>Daily Color</h2>
    <div class=\"note\">{daily_color_html}</div>
  </div>

  <div class=\"section\">
    <h2>FX Dashboard</h2>
    <div class=\"grid-3\">
      <div class=\"card\">
        <h3>G10 Spot</h3>
        {fx_g10_tbl}
      </div>
      <div class=\"card\">
        <h3>RMB Snapshot</h3>
        {fx_rmb_tbl}
        <div class=\"{spread_class}\">Spread Signal: {html_lib.escape(fx_spread_note)}</div>
      </div>
      <div class=\"card\">
        <h3>Rates &amp; Commodities</h3>
        {rates_comm_tbl}
      </div>
    </div>
    <div class=\"card\" style=\"margin-top:10px;\">
      <h3>USD/CNY Swap Points</h3>
      {fx_swap_tbl}
    </div>
  </div>

  <div class=\"section\">
    <h2>Macro Calendar</h2>
    {macro_html}
  </div>

  <div class=\"section\">
    <h2>US Close (Major Indices)</h2>
    {market_note_html}
    {us_eq_tbl}
    {chart_block(us_eq_chart_b64, us_eq_chart_conclusion)}
  </div>

  <div class=\"section\">
    <h2>US Rates (Treasury)</h2>
    <div class=\"grid-2\">
      <div class=\"card\">
        <h3>Latest Yields</h3>
        {ust_tbl}
        <h4>Key Spreads</h4>
        {ust_spreads_tbl}
        <div class=\"conclusion\">{html_lib.escape(ust_section_conclusion)}</div>
      </div>
      <div class=\"card\">
        <h3>Yield Curve</h3>
        {chart_block(ust_curve_b64, ust_curve_conclusion)}
      </div>
    </div>
  </div>

  <div class=\"section\">
    <h2>Rates Curve Visualizer</h2>
    <div class=\"grid-3\">
      <div class=\"card\">
        <h3>UST Curve Compare</h3>
        {chart_block(ust_curve_compare_b64, ust_curve_compare_conclusion)}
      </div>
      <div class=\"card\">
        <h3>UST 2s10s</h3>
        {chart_block(ust_2s10s_b64, ust_2s10s_conclusion)}
      </div>
      <div class=\"card\">
        <h3>UST 5s30s</h3>
        {chart_block(ust_5s30s_b64, ust_5s30s_conclusion)}
      </div>
    </div>
  </div>

  <div class=\"section\">
    <h2>Commodities</h2>
    <div class=\"grid-2\">
      <div class=\"card\">
        <h3>Snapshot</h3>
        {comm_tbl}
      </div>
      <div class=\"card\">
        <h3>Daily Moves</h3>
        {chart_block(comm_bar_b64, comm_bar_conclusion)}
      </div>
    </div>
  </div>

  <div class=\"section\">
    <h2>Cross-Asset</h2>
    <div class=\"grid-2\">
      <div class=\"card\">
        <h3>DXY vs Gold</h3>
        {corr_tbl}
        <div class=\"{corr_class}\">{html_lib.escape(corr_note)}</div>
      </div>
      <div class=\"card\">
        <h3>Rolling Corr (20D)</h3>
        {chart_block(corr_chart_b64, corr_chart_conclusion)}
      </div>
    </div>
  </div>

  <div class=\"section\">
    <h2>China Macro (PMI)</h2>
    {pmi_tbl}
    {chart_block(pmi_chart_b64, pmi_chart_conclusion)}
  </div>

  {foot_html}
</body>
</html>
"""


# ----------------------------
# Snapshot text for LLM
# ----------------------------


def build_snapshot_text(

    us_df: pd.DataFrame,
    ust_df: pd.DataFrame,
    comm_df: pd.DataFrame,
    pmi_df: pd.DataFrame,
    fx_g10_df: Optional[pd.DataFrame] = None,
    fx_rmb_df: Optional[pd.DataFrame] = None,
    corr_df: Optional[pd.DataFrame] = None,
) -> str:
    lines = []

    def df_to_lines(title: str, df: pd.DataFrame, max_rows=20) -> None:
        lines.append(f"{title}:")
        if df is None or df.empty:
            lines.append("  (no data)")
            return
        # plain text table-like
        cols = [k for k in df.columns if not str(k).endswith("Raw")]
        for _, r in df.head(max_rows).iterrows():
            pairs = [f"{k}={r[k]}" for k in cols]
            lines.append("  - " + ", ".join(pairs))

    df_to_lines("US Indices", us_df)
    df_to_lines("UST Yields", ust_df)
    df_to_lines("Commodities", comm_df)
    if fx_g10_df is not None:
        df_to_lines("FX G10", fx_g10_df)
    if fx_rmb_df is not None:
        df_to_lines("FX RMB", fx_rmb_df)
    df_to_lines("China PMI", pmi_df)
    if corr_df is not None and not corr_df.empty:
        latest = corr_df.iloc[-1]
        lines.append("Cross-Asset Correlation:")
        lines.append(
            "  - DXY_ret={:.4f}, Gold_ret={:.4f}, RollingCorr={:.2f}".format(
                latest.get("DXY_ret", float("nan")),
                latest.get("Gold_ret", float("nan")),
                latest.get("RollingCorr", float("nan")),
            )
        )

    return "\n".join(lines)




# ----------------------------
# Auto Conclusions
# ----------------------------

def _strip_tags(s: str) -> str:
    return re.sub(r"<[^>]+>", "", s or "")


def parse_numeric(x) -> float:
    """Parse numbers from strings like '+3.0', '3.59%', '+65.0 bp'."""
    if x is None:
        return float('nan')
    s = _strip_tags(str(x))
    s = s.replace(",", "")
    s = s.replace("%", "")
    s = s.replace("bp", "")
    s = s.strip()
    if not s or s.upper() == "N/A":
        return float('nan')
    try:
        return float(s)
    except Exception:
        return float('nan')


def _get_row(df: pd.DataFrame, col: str, value: str) -> Optional[pd.Series]:
    if df is None or df.empty or col not in df.columns:
        return None
    hit = df[df[col].astype(str) == value]
    if hit.empty:
        return None
    return hit.iloc[0]


def make_us_eq_chart_conclusion(us_eq_df: pd.DataFrame, us_market_note: str = "") -> str:
    if us_eq_df is None or us_eq_df.empty:
        return "Conclusion: US equity data unavailable."

    name_col = "Index" if "Index" in us_eq_df.columns else ("Asset" if "Asset" in us_eq_df.columns else us_eq_df.columns[0])

    def pct(candidates: List[str]) -> Tuple[float, str]:
        for name in candidates:
            r = _get_row(us_eq_df, name_col, name)
            if r is not None:
                return parse_numeric(r.get("Chg%")), name
        return float("nan"), candidates[0]

    spx_val, spx_label = pct(["S&P 500"])
    ndx_val, ndx_label = pct(["NASDAQ Comp", "Nasdaq 100", "Nasdaq"])
    dow_val, dow_label = pct(["Dow Jones", "Dow"])

    label_map = {
        "NASDAQ Comp": "Nasdaq",
        "Nasdaq 100": "Nasdaq 100",
        "Dow Jones": "Dow",
    }

    parts = []
    vals = []
    for label, v in [
        (label_map.get(spx_label, spx_label), spx_val),
        (label_map.get(ndx_label, ndx_label), ndx_val),
        (label_map.get(dow_label, dow_label), dow_val),
    ]:
        if not is_nan(v):
            parts.append(f"{label} {v:+.2f}%")
            vals.append(v)

    tone = "mixed tape"
    if vals:
        avg = sum(vals) / len(vals)
        if avg > 0.05:
            tone = "risk appetite improving"
        elif avg < -0.05:
            tone = "risk appetite softening"

    msg = f"Conclusion: {tone}; " + (", ".join(parts) if parts else "(major index moves unavailable)") + "."
    if us_market_note:
        msg += f" ({us_market_note})"
    return msg


def make_ust_section_conclusion(ust_latest_df: pd.DataFrame, ust_hist: pd.DataFrame) -> str:
    if ust_latest_df is None or ust_latest_df.empty:
        return "Conclusion: UST yield data unavailable."

    def yld(tenor: str) -> float:
        r = _get_row(ust_latest_df, "Tenor", tenor)
        return parse_numeric(r.get("Yield")) if r is not None else float('nan')

    def chg(tenor: str) -> float:
        r = _get_row(ust_latest_df, "Tenor", tenor)
        return parse_numeric(r.get("Δ(bp)")) if r is not None else float('nan')

    y2, dy2 = yld("2Y"), chg("2Y")
    y10, dy10 = yld("10Y"), chg("10Y")
    y30, dy30 = yld("30Y"), chg("30Y")

    sp = compute_key_spreads_bp(ust_hist)
    s10 = sp.get("2s10s", {})
    s53 = sp.get("5s30s", {})
    lvl_10 = s10.get("level_bp", float('nan'))
    chg_10 = s10.get("chg_bp", float('nan'))
    lvl_53 = s53.get("level_bp", float('nan'))
    chg_53 = s53.get("chg_bp", float('nan'))

    parts = []
    if not is_nan(y2):
        parts.append(f"2Y {y2:.2f}%({dy2:+.1f}bp)" if not is_nan(dy2) else f"2Y {y2:.2f}%")
    if not is_nan(y10):
        parts.append(f"10Y {y10:.2f}%({dy10:+.1f}bp)" if not is_nan(dy10) else f"10Y {y10:.2f}%")
    if not is_nan(y30):
        parts.append(f"30Y {y30:.2f}%({dy30:+.1f}bp)" if not is_nan(dy30) else f"30Y {y30:.2f}%")

    spread_bits = []
    if not is_nan(lvl_10):
        spread_bits.append(f"2s10s {lvl_10:+.1f}bp" + (f"({chg_10:+.1f}bp)" if not is_nan(chg_10) else ""))
    if not is_nan(lvl_53):
        spread_bits.append(f"5s30s {lvl_53:+.1f}bp" + (f"({chg_53:+.1f}bp)" if not is_nan(chg_53) else ""))

    shape = ""
    if not is_nan(dy2) and not is_nan(dy10):
        d_spread = dy10 - dy2
        if d_spread > 0.5:
            shape = "Curve steepening (long end up more)."
        elif d_spread < -0.5:
            shape = "Curve flattening (front end up more)."
        else:
            shape = "Curve shape broadly unchanged."

    msg = "Conclusion: " + (", ".join(parts) if parts else "insufficient tenor data")
    if spread_bits:
        msg += "; " + ", ".join(spread_bits)
    if shape:
        msg += " " + shape
    if not msg.endswith("."):
        msg += "."
    return msg


def make_ust_curve_conclusion(ust_hist: pd.DataFrame) -> str:
    if ust_hist is None or ust_hist.empty:
        return "Conclusion: curve data unavailable."

    df = ust_hist.dropna(subset=["Date"]).sort_values("Date")
    if df.empty:
        return "Conclusion: curve data unavailable."

    latest = df.iloc[-1]
    tenor_cols = [c for c in df.columns if c != "Date"]
    available = [c for c in tenor_cols if not is_nan(latest.get(c))]
    n = len(available)

    if n >= 10:
        return f"Conclusion: latest curve has {n} tenors (front to back), smoother for visualization."
    return f"Conclusion: latest curve has only {n} tenors (source limits/missing); consider alternate source or filling the short end."


def make_curve_compare_conclusion(ust_hist: pd.DataFrame, lookback_points: int = 5) -> str:
    if ust_hist is None or ust_hist.empty:
        return "Conclusion: history unavailable for comparison."
    df = ust_hist.dropna(subset=["Date"]).sort_values("Date")
    if len(df) < lookback_points + 1:
        return "Conclusion: insufficient history for a 1W comparison."
    latest = df.iloc[-1]
    prev = df.iloc[-(lookback_points + 1)]

    def bp(tenor: str) -> float:
        if tenor not in df.columns or is_nan(latest.get(tenor)) or is_nan(prev.get(tenor)):
            return float('nan')
        return (float(latest.get(tenor)) - float(prev.get(tenor))) * 100.0

    b2 = bp("2Y")
    b10 = bp("10Y")
    b30 = bp("30Y")

    parts = []
    if not is_nan(b2):
        parts.append(f"2Y {b2:+.0f}bp")
    if not is_nan(b10):
        parts.append(f"10Y {b10:+.0f}bp")
    if not is_nan(b30):
        parts.append(f"30Y {b30:+.0f}bp")

    shape = ""
    if not is_nan(b2) and not is_nan(b10):
        d_spread = b10 - b2
        if d_spread > 1:
            shape = "Curve steepened over the past week."
        elif d_spread < -1:
            shape = "Curve flattened over the past week."
        else:
            shape = "Curve shape changed only modestly over the past week."

    msg = f"Conclusion: over the past ~{lookback_points} sessions, " + (", ".join(parts) if parts else "curve changes not quantifiable") + "."
    if shape:
        msg += " " + shape
    return msg


def make_spread_conclusion(ust_hist: pd.DataFrame, long_t: str, short_t: str, lookback_points: int = 5) -> str:
    if ust_hist is None or ust_hist.empty:
        return "Conclusion: data unavailable."
    df = ust_hist.dropna(subset=["Date"]).sort_values("Date")
    if long_t not in df.columns or short_t not in df.columns:
        return "Conclusion: missing tenors; cannot compute."
    spread = (df[long_t] - df[short_t]) * 100.0
    spread = pd.to_numeric(spread, errors="coerce").dropna()
    if spread.empty:
        return "Conclusion: insufficient valid data."
    latest = float(spread.iloc[-1])

    msg = f"Conclusion: latest {latest:+.1f}bp"
    if len(spread) >= lookback_points + 1:
        prev = float(spread.iloc[-(lookback_points + 1)])
        msg += f", ~1W change {latest - prev:+.1f}bp."
    else:
        msg += "."
    return msg


def make_comm_bar_conclusion(comm_df: pd.DataFrame) -> str:
    if comm_df is None or comm_df.empty:
        return "Conclusion: commodities data unavailable."
    df = comm_df.copy()
    if "ChgPctRaw" in df.columns:
        df["ChgPctVal"] = pd.to_numeric(df["ChgPctRaw"], errors="coerce")
    elif "ChgPct" in df.columns:
        df["ChgPctVal"] = df["ChgPct"].map(parse_numeric)
    else:
        return "Conclusion: commodities data unavailable."
    df = df.dropna(subset=["ChgPctVal"])
    if df.empty:
        return "Conclusion: commodities data unavailable."
    top = df.iloc[df["ChgPctVal"].abs().values.argmax()]
    asset = top.get("Asset", top.get("Name", top.get("NameCN", "")))
    chg = float(top.get("ChgPctVal"))
    direction = "led gains" if chg > 0 else "led losses"
    return f"Conclusion: {asset} {direction} on the day ({chg:+.2f}%). Other commodities were relatively muted."


def make_corr_chart_conclusion(corr_df: pd.DataFrame) -> str:
    if corr_df is None or corr_df.empty:
        return "Conclusion: correlation data unavailable."
    last = corr_df.iloc[-1]
    c = last.get("RollingCorr")
    if is_nan(c):
        return "Conclusion: correlation data unavailable."
    c = float(c)
    if c > 0.3:
        desc = "positive correlation"
    elif c < -0.3:
        desc = "negative correlation"
    else:
        desc = "weak correlation"
    return f"Conclusion: 20D rolling correlation at {c:.2f} ({desc})."


def make_pmi_chart_conclusion(pmi_df: pd.DataFrame) -> str:
    if pmi_df is None or pmi_df.empty:
        return "Conclusion: PMI data unavailable."

    def actual(ind: str) -> float:
        r = _get_row(pmi_df, "Indicator", ind)
        return parse_numeric(r.get("Actual")) if r is not None else float('nan')

    mfg = actual("Official Mfg PMI")
    non = actual("Official Non-mfg PMI")
    cx = actual("Caixin Mfg PMI")

    bits = []
    if not is_nan(mfg):
        bits.append(f"Official Mfg PMI {mfg:.1f} ({'expansion' if mfg >= 50 else 'contraction'})")
    if not is_nan(non):
        bits.append(f"Official Non-mfg PMI {non:.1f} ({'expansion' if non >= 50 else 'contraction'})")
    if not is_nan(cx):
        bits.append(f"Caixin Mfg PMI {cx:.1f} ({'expansion' if cx >= 50 else 'contraction'})")

    if not bits:
        return "Conclusion: PMI data unavailable."

    return "Conclusion: " + ", ".join(bits) + "."
# ----------------------------
# Main
# ----------------------------

def main():
    load_dotenv(".env")

    out_path = Path("report.html").resolve()

    now = datetime.now()
    generated_at = now.strftime("%Y-%m-%d %H:%M:%S")

    # Fetch data
    us_eq_df, us_series, us_market_note = fetch_us_indices()
    ust_tbl_df, ust_hist = fetch_us_treasury_yields()
    comm_df = fetch_commodities_realtime()
    pmi_df, pmi_series = fetch_china_pmi_suite()
    fx_g10_df, fx_rmb_df, fx_swap_df, fx_meta = fetch_fx_dashboard()
    corr_df, corr_note = build_cross_asset_corr()
    macro_raw_df = fetch_macro_calendar_raw()
    macro_df, macro_note = generate_macro_calendar(macro_raw_df)

    # Charts
    us_chart = plot_us_indices_normalized(us_series)
    ust_curve = plot_ust_curve(ust_hist)
    ust_curve_compare = plot_ust_curve_compare(ust_hist)
    ust_2s10s = plot_2s10s_spread(ust_hist)
    ust_5s30s = plot_5s30s_spread(ust_hist)
    comm_bar = plot_commodities_bar(comm_df)
    pmi_chart = plot_pmi(pmi_series)
    corr_chart = plot_rolling_corr(corr_df)

    # LLM snapshot text
    snapshot_text = build_snapshot_text(
        us_eq_df,
        ust_tbl_df,
        comm_df,
        pmi_df,
        fx_g10_df=fx_g10_df,
        fx_rmb_df=fx_rmb_df,
        corr_df=corr_df,
    )
    daily_color = generate_daily_color(snapshot_text)

    # Render tables
    data_notes = "\n".join([
        "• UST curve: Treasury Daily Treasury Yield Curve Rates (CSV dataset).",
        "• Spreads: 2s10s = 10Y - 2Y; 5s30s = 30Y - 5Y (bp).",
        "• FX: spot and fixing data mainly from AKShare (EastMoney/ChinaMoney). N/A usually means source missing or temporarily unavailable.",
    ])

    comm_tbl_df = comm_df
    if comm_df is not None and not comm_df.empty:
        cols = [c for c in ["Asset", "Last", "Chg", "ChgPct", "Time", "Date"] if c in comm_df.columns]
        comm_tbl_df = comm_df[cols].copy()

    pmi_tbl_df = mark_delayed_if_missing(pmi_df, label="Delayed Data")
    rates_comm_df = build_rates_commodities_table(ust_tbl_df, comm_df, ust_hist=ust_hist)
    macro_html = render_macro_calendar_html(macro_df, macro_note)

    # UST key spreads table
    ust_spreads_df = build_spreads_table(ust_hist)
    ust_spreads_tbl_html = (
        df_to_html_table(ust_spreads_df, colorize_cols=["Level (bp)", "Δ (bp)"])
        if (ust_spreads_df is not None and not ust_spreads_df.empty)
        else "<div class='muted'>N/A</div>"
    )

    # Cross-asset snapshot table (latest)
    corr_tbl_df = pd.DataFrame()
    if corr_df is not None and not corr_df.empty:
        last = corr_df.iloc[-1]
        dxy_ret = last.get("DXY_ret")
        gold_ret = last.get("Gold_ret")
        corr_tbl_df = pd.DataFrame([
            {
                "Date": last.name.date().isoformat() if hasattr(last.name, "date") else str(last.name),
                "DXY": fmt_num(last.get("DXY"), 2),
                "Gold": fmt_num(last.get("Gold"), 2),
                "DXY Ret": fmt_signed(dxy_ret * 100.0, 2, "%") if not is_nan(dxy_ret) else "N/A",
                "Gold Ret": fmt_signed(gold_ret * 100.0, 2, "%") if not is_nan(gold_ret) else "N/A",
                "Rolling Corr": fmt_num(last.get("RollingCorr"), 2),
            }
        ])

    # Auto conclusions
    us_eq_chart_conclusion = make_us_eq_chart_conclusion(us_eq_df, us_market_note)
    ust_section_conclusion = make_ust_section_conclusion(ust_tbl_df, ust_hist)
    ust_curve_conclusion = make_ust_curve_conclusion(ust_hist)
    ust_curve_compare_conclusion = make_curve_compare_conclusion(ust_hist)
    ust_2s10s_conclusion = make_spread_conclusion(ust_hist, "10Y", "2Y")
    ust_5s30s_conclusion = make_spread_conclusion(ust_hist, "30Y", "5Y")
    comm_bar_conclusion = make_comm_bar_conclusion(comm_df)
    corr_chart_conclusion = make_corr_chart_conclusion(corr_df)
    pmi_chart_conclusion = make_pmi_chart_conclusion(pmi_df)

    html_out = render_html(
        generated_at=generated_at,
        daily_color=daily_color,
        us_eq_tbl=df_to_html_table(us_eq_df),
        us_eq_chart_b64=us_chart,
        us_eq_chart_conclusion=us_eq_chart_conclusion,
        us_market_note=us_market_note,
        fx_g10_tbl=df_to_html_table(fx_g10_df),
        fx_rmb_tbl=df_to_html_table(fx_rmb_df, colorize_cols=["Chg", "Chg%"]),
        fx_spread_note=fx_meta.get("spread_note", "N/A"),
        fx_swap_tbl=df_to_html_table(fx_swap_df, colorize_cols=[]),
        rates_comm_tbl=df_to_html_table(rates_comm_df, colorize_cols=["Chg (bp/%)"]),
        ust_tbl=df_to_html_table(ust_tbl_df),
        ust_spreads_tbl=ust_spreads_tbl_html,
        ust_section_conclusion=ust_section_conclusion,
        ust_curve_b64=ust_curve,
        ust_curve_conclusion=ust_curve_conclusion,
        ust_curve_compare_b64=ust_curve_compare,
        ust_curve_compare_conclusion=ust_curve_compare_conclusion,
        ust_2s10s_b64=ust_2s10s,
        ust_2s10s_conclusion=ust_2s10s_conclusion,
        ust_5s30s_b64=ust_5s30s,
        ust_5s30s_conclusion=ust_5s30s_conclusion,
        comm_tbl=df_to_html_table(comm_tbl_df),
        comm_bar_b64=comm_bar,
        comm_bar_conclusion=comm_bar_conclusion,
        pmi_tbl=df_to_html_table(pmi_tbl_df),
        pmi_chart_b64=pmi_chart,
        pmi_chart_conclusion=pmi_chart_conclusion,
        corr_tbl=df_to_html_table(corr_tbl_df, colorize_cols=["DXY Ret", "Gold Ret", "Rolling Corr"]),
        corr_chart_b64=corr_chart,
        corr_chart_conclusion=corr_chart_conclusion,
        corr_note=corr_note,
        macro_html=macro_html,
        data_notes=data_notes,
    )

    out_path.write_text(html_out, encoding="utf-8")
    print(f"[OK] Report generated: {out_path}")

    # Auto open in browser
    try:
        webbrowser.open(out_path.as_uri())
    except Exception:
        pass


if __name__ == "__main__":

    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(130)
    except Exception as e:
        print(f"[ERROR] {type(e).__name__}: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)
