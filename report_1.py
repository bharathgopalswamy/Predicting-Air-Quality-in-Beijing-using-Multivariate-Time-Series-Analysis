import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Team Members:
# 1.Bharath Gopalsamy
# 2.Lokesh Janakiraman

# This code is for Report 1 second phase (preliminary analysis + visualization).
# It reads the subset CSV file path from the command line, prints ANALYSIS metrics,
# and then shows 5 different plots.


def infer_datetime(df: pd.DataFrame):
    # Infering datetime from common columns (year/month/day/hour or a datetime-like column)
    lower = {str(c).lower(): c for c in df.columns}

    # single datetime-like column
    for k in ["datetime", "timestamp", "date", "time"]:
        if k in lower:
            dt = pd.to_datetime(df[lower[k]], errors="coerce")
            if dt.notna().sum() > 0:
                return dt

    # split the columns
    if all(k in lower for k in ["year", "month", "day"]):
        y = df[lower["year"]]
        m = df[lower["month"]]
        d = df[lower["day"]]
        h = df[lower["hour"]] if "hour" in lower else 0
        dt = pd.to_datetime(dict(year=y, month=m, day=d, hour=h), errors="coerce")
        if dt.notna().sum() > 0:
            return dt

    return None


def safe_mode(series: pd.Series):
    #Computing a safe mode that returns 'NA' if the series has no valid values
    s = series.dropna()
    if s.empty:
        return "NA"
    m = s.mode()
    if m.empty:
        return "NA"
    return m.iloc[0]


def fmt(x):
    # Formating values for clean console output (handles NA, floats, ints, and strings)
    if x is None:
        return "NA"
    if isinstance(x, str):
        return x
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    if isinstance(x, (float, np.floating)):
        if np.isnan(x):
            return "NA"
        return f"{float(x):.4g}"
    return str(x)


def pick_columns(df: pd.DataFrame):

#Detecting numeric and categorical columns; remove index-like/non-meaningful numeric columns.
    
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    numeric_cols = [c for c in numeric_cols if c.lower() not in {"no", "index", "wd"}]


    from pandas.api.types import CategoricalDtype

    cat_cols = []
    for c in df.columns:
        dt = df[c].dtype
        if dt == "object" or isinstance(dt, CategoricalDtype):
            cat_cols.append(c)

    return numeric_cols, cat_cols



def compute_metrics(df: pd.DataFrame, numeric_cols):
    """
    Required: Range, Mean, Mode where it makes sense.
    Extra metrics (2): StdDev and Skewness (numeric only).
    """
    cols = [c for c in df.columns if c.lower() not in {"no", "index"}]


    # Range (numeric only)
    range_row = []
    for c in cols:
        if c in numeric_cols:
            s = df[c].dropna()
            if s.empty:
                range_row.append("NA")
            else:
                range_row.append(f"[{fmt(s.min())},{fmt(s.max())}]")
        else:
            range_row.append("NA")

    # Mean (numeric only)
    mean_row = []
    for c in cols:
        mean_row.append(df[c].mean(skipna=True) if c in numeric_cols else "NA")

    # Mode (works for both numeric and categorical columns)
    mode_row = [safe_mode(df[c]) for c in cols]

    # Extra metric A: StdDev (numeric only)
    std_row = []
    for c in cols:
        std_row.append(df[c].std(skipna=True) if c in numeric_cols else "NA")

    # Extra metric B: Skewness (numeric only)
    skew_row = []
    for c in cols:
        skew_row.append(df[c].skew(skipna=True) if c in numeric_cols else "NA")
 # Organizing all metric rows in the required output format
    rows = {
        "Range": range_row,
        "Mean": mean_row,
        "Mode": mode_row,
        "M_a(StdDev)": std_row,
        "M_b(Skew)": skew_row,
    }
    return cols, rows


def print_required(cols, rows):
# Print: ANALYSIS + metrics table.
    print("ANALYSIS")
    print("\t".join([str(c) for c in cols]))
    for metric_name, values in rows.items():
        print(metric_name + "\t" + "\t".join([fmt(v) if fmt(v) != "" else "NA" for v in values]))



def make_5_figures(df: pd.DataFrame, numeric_cols, cat_cols):
   
    # If no numeric columns, still show 5 figures without crashing
    if len(numeric_cols) == 0:
        for i in range(5):
            plt.figure()
            plt.title(f"Figure {i+1}")
            plt.text(0.5, 0.5, "No numeric columns available", ha="center", va="center")
        plt.show()
        return

    num1 = numeric_cols[0]
    num2 = numeric_cols[1] if len(numeric_cols) > 1 else numeric_cols[0]

    # Prefer "PM2.5" and "PM10" if present
    if "PM2.5" in df.columns and "PM10" in df.columns:
        num1, num2 = "PM2.5", "PM10"
    elif "PM2.5" in df.columns:
        num1 = "PM2.5"

    # Prefer "station" as grouping if present
    group = "station" if "station" in df.columns else (cat_cols[0] if len(cat_cols) > 0 else None)

    # 1. Histogram
    plt.figure()
    df[num1].dropna().hist(bins=30)
    plt.title(f"Histogram of {num1}")
    plt.xlabel(num1)
    plt.ylabel("Frequency")

    # 2. Boxplot (grouped if possible)
    plt.figure()
    if group is not None:
        tmp = df[[group, num1]].dropna()
        top = tmp[group].value_counts().head(10).index  # keep readable
        tmp = tmp[tmp[group].isin(top)]
        data = [tmp[tmp[group] == g][num1].values for g in top]
        plt.boxplot(data, tick_labels=[str(g) for g in top], showfliers=True)
        plt.title(f"Boxplot of {num1} by {group} (top 10)")
        plt.xlabel(group)
        plt.ylabel(num1)
        plt.xticks(rotation=45, ha="right")
    else:
        plt.boxplot(df[num1].dropna().values, showfliers=True)
        plt.title(f"Boxplot of {num1}")
        plt.ylabel(num1)

    # 3. Scatter plot
    plt.figure()
    plt.scatter(df[num1], df[num2], alpha=0.7)
    plt.title(f"Scatter: {num1} vs {num2}")
    plt.xlabel(num1)
    plt.ylabel(num2)

    # 4. Line plot (datetime if possible, otherwise index plot)
    plt.figure()
    dt = infer_datetime(df)
    if dt is not None:
        tmp = df.copy()
        tmp["_dt_"] = dt
        tmp = tmp.dropna(subset=["_dt_", num1]).sort_values("_dt_")
        plt.plot(tmp["_dt_"], tmp[num1])
        plt.title(f"Time series of {num1}")
        plt.xlabel("Time")
        plt.ylabel(num1)
        plt.xticks(rotation=45, ha="right")
    else:
        series = df[num1].dropna()
        plt.plot(series.values)
        plt.title(f"Index plot of {num1} (no datetime detected)")
        plt.xlabel("Row index (subset)")
        plt.ylabel(num1)

    # 5. Correlation heatmap
    plt.figure()
    num_df = df[numeric_cols].dropna(axis=1, how="all")
    if num_df.shape[1] >= 2:
        corr = num_df.corr(numeric_only=True)
        plt.imshow(corr.values, aspect="auto")
        plt.title("Correlation Heatmap (numeric columns)")
        plt.xticks(range(corr.shape[1]), [str(c) for c in corr.columns], rotation=90)
        plt.yticks(range(corr.shape[0]), [str(c) for c in corr.index])
        plt.colorbar()
    else:
        plt.title("Correlation Heatmap")
        plt.text(0.5, 0.5, "Not enough numeric columns", ha="center", va="center")

    plt.tight_layout()
    plt.show()


def main():
    
    if len(sys.argv) != 2:
       
        sys.exit(1)

    subset_path = sys.argv[1]
    df = pd.read_csv(subset_path)

    numeric_cols, cat_cols = pick_columns(df)
    cols, rows = compute_metrics(df, numeric_cols)

    
    print_required(cols, rows)

   
    make_5_figures(df, numeric_cols, cat_cols)


if __name__ == "__main__":
    main()
