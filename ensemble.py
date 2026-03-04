"""
Ensemble 3 file predictions + character-level voting + length fix + post-processing.

Step 1: Character-level ensemble — vote từng ký tự theo position
Step 2: Sửa length sai (6 hoặc 8 ký tự → thử về 7)
Step 3: Post-processing format Brazil/Mercosur
"""
import re
import os
from collections import defaultdict

# ── Config ───────────────────────────────────────────────────
FILES = [
    ("/home/fluuvys/ICPR Competition/ICPR_2026_Final_Solution/results/final_predictions/submission_mamba_only.txt", 1.0),
    ("/home/fluuvys/ICPR Competition/ICPR_2026_Final_Solution/results/final_predictions/submission_new_svtr_final.txt",   1.0),
    ("/home/fluuvys/ICPR Competition/ICPR_2026_Final_Solution/results/final_predictions/submission_restran_only.txt",    1.0),
    ("/home/fluuvys/ICPR Competition/ICPR_2026_Final_Solution/results/final_predictions/submission_svtr_only.txt", 1.0),
    ("/home/fluuvys/ICPR Competition/ICPR_2026_Final_Solution/results/final_predictions/submission_new_restran_final.txt", 1.0)
]
OUTPUT = "/home/fluuvys/ICPR Competition/ICPR_2026_Final_Solution/results/ensemble_results/submission_ensemble_v2.txt"

# ── OCR Confusion table ───────────────────────────────────────
CHAR_CONFUSION = {
    "O": "0",  "0": "O",
    "I": "1",  "1": "I",
    "Z": "2",  "2": "Z",
    "S": "5",  "5": "S",
    "G": "6",  "6": "G",
    "B": "8",  "8": "B",
    "Q": "0",
    "D": "0",
}
CONFUSION_ALPHA = {k: v for k, v in CHAR_CONFUSION.items() if v.isalpha()}
CONFUSION_DIGIT = {k: v for k, v in CHAR_CONFUSION.items() if v.isdigit()}

# ── Format patterns ───────────────────────────────────────────
BRAZIL_RE   = re.compile(r'^[A-Z]{3}[0-9]{4}$')
MERCOSUR_RE = re.compile(r'^[A-Z]{3}[0-9]{1}[A-Z]{1}[0-9]{2}$')

def is_valid(text: str) -> bool:
    return bool(BRAZIL_RE.match(text) or MERCOSUR_RE.match(text))

def detect_format(text: str) -> str:
    if len(text) != 7:
        return "unknown"
    return "mercosur" if text[4].isalpha() else "brazil"

def expected_type(pos: int, fmt: str) -> str:
    if fmt == "brazil":
        return "alpha" if pos < 3 else "digit"
    elif fmt == "mercosur":
        return "alpha" if pos in (0, 1, 2, 4) else "digit"
    return "any"

def try_fix_format(text: str) -> str:
    """Sửa ký tự sai type dựa vào confusion table."""
    if len(text) != 7:
        return text
    fmt = detect_format(text)
    if fmt == "unknown":
        return text
    fixed = list(text)
    for i, ch in enumerate(fixed):
        expected = expected_type(i, fmt)
        if expected == "digit" and not ch.isdigit():
            candidate = CHAR_CONFUSION.get(ch)
            if candidate and candidate.isdigit():
                fixed[i] = candidate
        elif expected == "alpha" and not ch.isalpha():
            candidate = CHAR_CONFUSION.get(ch)
            if candidate and candidate.isalpha():
                fixed[i] = candidate
    return "".join(fixed)

# ════════════════════════════════════════════════════════════
# STEP 1: LOAD + CHARACTER-LEVEL VOTING
# char_votes[track_id][position][char] += weight * confidence
# string_votes[track_id][text] += weight * confidence (fallback)
# ════════════════════════════════════════════════════════════
char_votes   = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
string_votes = defaultdict(lambda: defaultdict(float))
conf_map     = defaultdict(lambda: defaultdict(float))

for fpath, weight in FILES:
    if not os.path.exists(fpath):
        print(f"⚠️  Không tìm thấy: {fpath} — bỏ qua")
        continue
    count = 0
    with open(fpath, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                track_part, conf_part = line.rsplit(";", 1)
                track_id, text        = track_part.split(",", 1)
                conf                  = float(conf_part)
            except Exception:
                try:
                    track_id, text = line.split(",", 1)
                    conf = 1.0
                except Exception:
                    continue

            text = text.upper().strip()
            string_votes[track_id][text] += weight * conf

            # Character-level vote chỉ cho text đúng 7 ký tự
            if len(text) == 7:
                for pos, ch in enumerate(text):
                    char_votes[track_id][pos][ch] += weight * conf

            count += 1
    print(f"✅ {os.path.basename(fpath)}: {count} predictions")

print(f"\n📊 Total tracks: {len(string_votes)}")

# ════════════════════════════════════════════════════════════
# STEP 2: BUILD BEST TEXT PER TRACK
# ════════════════════════════════════════════════════════════

def char_level_best(track_id: str) -> tuple:
    """Ghép ký tự win nhất từng position → best text + avg confidence."""
    cvotes = char_votes[track_id]
    if not cvotes or len(cvotes) != 7:
        return None, 0.0
    chars = []
    confs = []
    for pos in range(7):
        if not cvotes[pos]:
            return None, 0.0
        best_ch    = max(cvotes[pos], key=cvotes[pos].get)
        total_pos  = sum(cvotes[pos].values())
        char_conf  = cvotes[pos][best_ch] / total_pos
        chars.append(best_ch)
        confs.append(char_conf)
    return "".join(chars), sum(confs) / len(confs)

def string_level_best(track_id: str) -> tuple:
    """Fallback: majority vote cả chuỗi."""
    svotes     = string_votes[track_id]
    best_text  = max(svotes, key=svotes.get)
    total      = sum(svotes.values())
    conf       = svotes[best_text] / total if total > 0 else 0.0
    return best_text, conf

# ════════════════════════════════════════════════════════════
# STEP 3: LENGTH FIX — sửa text 6 hoặc 8 ký tự → 7
# ════════════════════════════════════════════════════════════

def levenshtein(a: str, b: str) -> int:
    """Tính edit distance giữa 2 chuỗi."""
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            temp = dp[j]
            if a[i-1] == b[j-1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j-1])
            prev = temp
    return dp[n]

def generate_valid_candidates(text: str) -> list:
    """
    Sinh tất cả biển số hợp lệ trong khoảng edit distance <= 2 từ text.
    Chiến lược: thử substitute từng ký tự theo confusion table + brute force
    1 substitution trên toàn bộ charset.
    """
    candidates = []
    ALPHA = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    DIGIT = "0123456789"

    # Pad hoặc trim về 7 trước
    t = text[:7].ljust(7, "0") if len(text) < 7 else text[:7]

    # 1 substitution: thử thay từng ký tự
    for pos in range(7):
        for ch in (ALPHA + DIGIT):
            candidate = t[:pos] + ch + t[pos+1:]
            if is_valid(candidate):
                candidates.append((candidate, levenshtein(text, candidate)))

    # 1 insertion + 1 deletion đã handle trong try_fix_length
    # Thêm: 2 substitutions (chỉ theo confusion table để giới hạn search)
    for pos1 in range(7):
        ch1 = CHAR_CONFUSION.get(t[pos1])
        if not ch1:
            continue
        t1 = t[:pos1] + ch1 + t[pos1+1:]
        for pos2 in range(pos1+1, 7):
            ch2 = CHAR_CONFUSION.get(t1[pos2])
            if not ch2:
                continue
            candidate = t1[:pos2] + ch2 + t1[pos2+1:]
            if is_valid(candidate):
                candidates.append((candidate, levenshtein(text, candidate)))

    return candidates

def levenshtein_fix(text: str) -> str:
    """
    Tìm biển số hợp lệ gần nhất (edit distance nhỏ nhất) với text.
    Chỉ accept nếu distance <= 2.
    """
    candidates = generate_valid_candidates(text)
    if not candidates:
        return text
    # Chọn candidate có distance nhỏ nhất
    best = min(candidates, key=lambda x: x[1])
    if best[1] <= 2:
        return best[0]
    return text

def try_fix_length(text: str) -> str:
    """
    Thử sửa text về 7 ký tự:
    - 6 ký tự: thử insert ký tự hợp lý vào từng position
    - 8 ký tự: thử xóa ký tự trùng lặp hoặc thừa
    """
    if len(text) == 6:
        # Thử insert vào từng position, kiểm tra format hợp lệ
        for pos in range(7):
            # Insert digit
            for ch in "0123456789":
                candidate = text[:pos] + ch + text[pos:]
                if is_valid(candidate):
                    return candidate
            # Insert alpha
            for ch in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
                candidate = text[:pos] + ch + text[pos:]
                if is_valid(candidate):
                    return candidate
    elif len(text) == 8:
        # Thử xóa từng ký tự
        for pos in range(8):
            candidate = text[:pos] + text[pos+1:]
            if is_valid(candidate):
                return candidate
    return text

# ════════════════════════════════════════════════════════════
# MAIN: combine tất cả steps
# ════════════════════════════════════════════════════════════
results = []
stats   = defaultdict(int)

for track_id in sorted(string_votes.keys()):
    # Ưu tiên character-level ensemble nếu có đủ 7 ký tự
    char_text, char_conf = char_level_best(track_id)
    str_text,  str_conf  = string_level_best(track_id)

    # Chọn text tốt hơn: ưu tiên char-level nếu hợp lệ hơn
    if char_text and is_valid(char_text):
        best_text, conf = char_text, char_conf
        stats["char_valid"] += 1
    elif is_valid(str_text):
        best_text, conf = str_text, str_conf
        stats["string_valid"] += 1
    else:
        # Thử fix length
        candidate = char_text if char_text else str_text
        conf      = char_conf if char_text else str_conf
        length_fixed = try_fix_length(candidate)

        if is_valid(length_fixed):
            best_text = length_fixed
            stats["length_fixed"] += 1
        else:
            # Thử fix format (char confusion)
            format_fixed = try_fix_format(candidate)
            if is_valid(format_fixed):
                best_text = format_fixed
                stats["format_fixed"] += 1
            else:
                # Thử Levenshtein fix — tìm biển số hợp lệ gần nhất
                lev_fixed = levenshtein_fix(candidate)
                if is_valid(lev_fixed):
                    best_text = lev_fixed
                    stats["lev_fixed"] += 1
                else:
                    best_text = candidate
                    stats["unfixed"] += 1

    results.append(f"{track_id},{best_text};{conf:.4f}")

# ── Save ─────────────────────────────────────────────────────
os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
with open(OUTPUT, "w") as f:
    f.write("\n".join(results))

print(f"\n📊 Kết quả:")
print(f"   ✅ Char-level valid  : {stats['char_valid']}")
print(f"   ✅ String-level valid: {stats['string_valid']}")
print(f"   📏 Sửa được length  : {stats['length_fixed']}")
print(f"   🔧 Sửa được format  : {stats['format_fixed']}")
print(f"   📐 Sửa được lev     : {stats['lev_fixed']}")
print(f"   ❌ Không sửa được   : {stats['unfixed']}")
print(f"\n💾 Đã lưu {len(results)} predictions → {OUTPUT}")