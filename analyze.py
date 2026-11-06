import json
import unicodedata
import os
from collections import defaultdict

# ---------- Load Records (Compatible with JSON Array / Single Object / JSONL) ----------
def load_records(path):
    with open(path, "r", encoding="utf-8") as f:
        content = f.read().strip()
    try:
        data = json.loads(content)
        if isinstance(data, dict):
            data = [data]
        return data
    except json.JSONDecodeError:
        return [json.loads(line) for line in content.splitlines() if line.strip()]


# ---------- Find the 1-based position of nl_input.url in code_input, return None if not found ----------
def find_rank(rec):
    target = rec["nl_input"]["url"]
    for i, ci in enumerate(rec["nl_input"].get("code_input", [])):
        if ci.get("url") == target:
            return i + 1
    return None


# ---------- Matching Key: Identify identical records by the natural language text in nl_input ----------
def rec_key(rec):
    text = rec["nl_input"].get("nl_input", "")
    return " ".join(text.split()).lower()


# ---------- Calculate hit rate and top1/top3/top10 statistics for a set of records ----------
def rank_stats(records):
    total = len(records)
    hit = top1 = top3 = top10 = 0
    for r in records:
        rk = find_rank(r)
        if rk is None:
            continue
        hit += 1
        if rk == 1:
            top1 += 1
        if rk <= 3:
            top3 += 1
        if rk <= 10:
            top10 += 1
    return {"total": total, "hit": hit, "top1": top1,
            "top3": top3, "top10": top10, "miss": total - hit}


# ---------- Table Formatting Utilities ----------
def str_width(s):
    return sum(2 if unicodedata.east_asian_width(c) in ("W", "F") else 1
               for c in s)


def pad(s, width):
    return s + " " * (width - str_width(s))


def ratio(num, den):
    if den == 0:
        return f"{num}/{den} (N/A)"
    return f"{num}/{den} ({num / den * 100:.2f}%)"


def print_table(title, headers, rows):
    print(f"\n{title}")
    widths = [str_width(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], str_width(cell))
    sep = "+" + "+".join("-" * (w + 2) for w in widths) + "+"
    print(sep)
    print("|" + "|".join(" " + pad(h, widths[i]) + " "
                         for i, h in enumerate(headers)) + "|")
    print(sep)
    for row in rows:
        print("|" + "|".join(" " + pad(c, widths[i]) + " "
                             for i, c in enumerate(row)) + "|")
    print(sep)


# ---------- Main Execution Flow ----------
def analyze(FILE1, FILE2):
    # Extract clean filenames from full paths for display
    name1 = os.path.basename(FILE1)
    name2 = os.path.basename(FILE2)

    data1 = load_records(FILE1)
    data2 = load_records(FILE2)

    # Build a 1-to-1 index for file2 using nl_input text as the key
    idx2 = {}
    dup2 = 0
    for r in data2:
        k = rec_key(r)
        if k in idx2:
            dup2 += 1
            continue
        idx2[k] = r

    # ================= [A] Independent Statistics for Each File =================
    s1 = rank_stats(data1)
    s2 = rank_stats(data2)

    rows_a = [
        ["URL exists in code_input",  ratio(s1["hit"],   s1["total"]),
                                      ratio(s2["hit"],   s2["total"])],
        ["  ├─ Top 1",                ratio(s1["top1"],  s1["total"]),
                                      ratio(s2["top1"],  s2["total"])],
        ["  ├─ Top 3",                ratio(s1["top3"],  s1["total"]),
                                      ratio(s2["top3"],  s2["total"])],
        ["  └─ Top 10",               ratio(s1["top10"], s1["total"]),
                                      ratio(s2["top10"], s2["total"])],
        ["URL not in code_input",     ratio(s1["miss"],  s1["total"]),
                                      ratio(s2["miss"],  s2["total"])],
        ["Total Records",            str(s1["total"]),  str(s2["total"])],
    ]
    print("\n" + "=" * 78)
    print(f"[A] Independent Statistics for Each File (denominator = total records in the file)")
    print_table("", ["Metric", name1, name2], rows_a)

    # ================= [B] Cross Analysis (1-to-1 Matching by nl_input Text) =================
    g_hit  = [r for r in data1 if find_rank(r) is not None]   # Records in file1 where URL is inside code_input
    g_miss = [r for r in data1 if find_rank(r) is None]       # Records in file1 where URL is NOT inside code_input

    def map_and_analyze(subset):
        """Map file1 records to file2 unique records 1-to-1 by nl_input text,
        then calculate the rank distribution of those corresponding file2 records in their own code_input list."""
        matched, no_match = [], 0
        for r1 in subset:
            r2 = idx2.get(rec_key(r1))
            if r2 is None:
                no_match += 1
            else:
                matched.append(r2)
        st = rank_stats(matched)
        st["no_match"]    = no_match
        st["file1_count"] = len(subset)
        return st

    st_hit  = map_and_analyze(g_hit)
    st_miss = map_and_analyze(g_miss)

    rows_b = [
        # ---- Group 1: Records in file1 where URL exists in code_input ----
        [f"{name1} group: URL in code_input",
         str(st_hit["file1_count"]),
         str(st_hit["total"]),
         str(st_hit["no_match"]),
         ratio(st_hit["hit"],  st_hit["total"]),          # Denominator = corresponding file2 records
         ratio(st_hit["top1"],  st_hit["file1_count"]),   # Denominator = all records in this file1 group
         ratio(st_hit["top3"],  st_hit["file1_count"]),
         ratio(st_hit["top10"], st_hit["file1_count"])],
        # ---- Group 2: Records in file1 where URL does NOT exist in code_input ----
        [f"{name1} group: URL not in code_input",
         str(st_miss["file1_count"]),
         str(st_miss["total"]),
         str(st_miss["no_match"]),
         ratio(st_miss["hit"],  st_miss["total"]),        # Denominator = corresponding file2 records
         ratio(st_miss["top1"],  st_miss["file1_count"]), # Denominator = all records in this file1 group
         ratio(st_miss["top3"],  st_miss["file1_count"]),
         ratio(st_miss["top10"], st_miss["file1_count"])],
    ]

    print("\n" + "=" * 78)
    print(f"[B] Cross Analysis ({name1} → {name2}, 1-to-1 matching by nl_input text)")
    print(f"    For each subset of {name1} records, match to {name2} records by nl_input text, then calculate:")
    print(f"      · Whether nl_input.url exists in code_input in {name2}   → denominator = corresponding {name2} records")
    print(f"      · If exists, its rank in top 1 / top 3 / top 10        → denominator = total records in this {name1} group")
    print(f"        (Group 1 denominator = all 'in code' records; Group 2 denominator = all 'not in code' records)")
    print_table(
        "",
        [f"{name1} Group", f"{name1} Count", f"{name2} Matched", "No Match",
         f"{name2} Hit", "Top 1", "Top 3", "Top 10"],
        rows_b,
    )

def main():
    solidity1 = "./MMCroCS/result/solidity_res3.jsonl"
    solidity2 = "./MMCroCS/result/solidity_res10_ex.jsonl"
    cosqa1 = "./MMCroCS/result/cosqa_res3.jsonl"
    cosqa2 = "./MMCroCS/result/cosqa_res10_ex.jsonl"
    sql1 = "./MMCroCS/result/sql_res3.jsonl"
    sql2 = "./MMCroCS/result/sql_res10_ex.jsonl"
    analyze(sql1, sql2)
    analyze(cosqa1, cosqa2)
    analyze(solidity1, solidity2)

if __name__ == "__main__":
    main()
