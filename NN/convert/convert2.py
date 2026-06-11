import re
from pathlib import Path

import pandas as pd


# ==========================================
# 1️⃣ INTERACTIVE HELPERS
# ==========================================

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_FILE = "gozaresh.xlsx"
DEFAULT_OUTPUT_FILE = "sugar_all_days_clean_7.csv"


def ask_text(prompt, default=None):
    """Ask for text input, returning the default when the user presses Enter."""
    suffix = f" [{default}]" if default else ""
    value = input(f"{prompt}{suffix}: ").strip()
    return value or default


def ask_yes_no(prompt, default=False):
    """Ask a yes/no question and return True for yes, False for no."""
    default_text = "Y/n" if default else "y/N"

    while True:
        answer = input(f"{prompt} ({default_text}): ").strip().lower()
        if not answer:
            return default
        if answer in {"y", "yes"}:
            return True
        if answer in {"n", "no"}:
            return False
        print("Please answer with y/yes or n/no.")


def resolve_existing_path(file_path):
    """Return an existing path from the current directory or this script directory."""
    path = Path(file_path)
    if path.exists() or path.is_absolute():
        return path

    script_relative_path = SCRIPT_DIR / path
    if script_relative_path.exists():
        return script_relative_path

    return path


def ask_csv_files(prompt):
    """Ask for one or more CSV files as a comma-separated list."""
    while True:
        raw_files = input(f"{prompt}: ").strip()
        files = [resolve_existing_path(item.strip()) for item in raw_files.split(",") if item.strip()]

        if not files:
            print("Please enter at least one CSV file path.")
            continue

        missing_files = [str(file_path) for file_path in files if not file_path.exists()]
        if missing_files:
            print("These CSV files were not found:")
            for missing_file in missing_files:
                print(f"  - {missing_file}")
            continue

        return files


# ==========================================
# 2️⃣ CLEAN NUMBER FUNCTION
# ==========================================


def clean_number(x):
    if pd.isna(x):
        return None

    x = str(x)

    # Persian → English digits
    persian_digits = "۰۱۲۳۴۵۶۷۸۹"
    english_digits = "0123456789"
    for p, e in zip(persian_digits, english_digits):
        x = x.replace(p, e)

    x = x.replace(",", "")
    x = x.replace("٫", ".")
    x = x.replace("...", "")
    x = x.strip()

    try:
        return float(x)
    except ValueError:
        return None


def clean_cell_value(x):
    """Return a spreadsheet cell as a useful CSV value, keeping text ranges like 10-12."""
    if pd.isna(x):
        return None

    text = str(x).strip()
    if not text or text.lower() == "nan" or text in {"...", "…"}:
        return None

    number = clean_number(text)
    if number is not None:
        return number

    return text


# ==========================================
# 3️⃣ SECTION + PROPERTY FINDER
# ==========================================


def extract_number_near_property(df, row_idx, col_idx, max_row_lookahead=2, max_col_offset=2):
    """
    Extract a number that is most likely associated with a property label cell.
    Priority:
      1) Same cell
      2) Same column in next rows (for header-above-value layouts, e.g. white_* section)
      3) Nearby cells in same row (small horizontal window only)
      4) Nearby columns in next rows (small horizontal window)
    """

    current_row = df.iloc[row_idx]

    # 1) Same cell
    num = clean_number(current_row.iloc[col_idx])
    if num is not None:
        return num

    # 2) Same column in next rows
    for r in range(row_idx + 1, min(row_idx + 1 + max_row_lookahead, len(df))):
        next_row = df.iloc[r]
        num = clean_number(next_row.iloc[col_idx])
        if num is not None:
            return num

    # 3) Nearby cells in same row (closest first)
    for offset in range(1, max_col_offset + 1):
        right = col_idx + offset
        left = col_idx - offset

        if right < len(current_row):
            num = clean_number(current_row.iloc[right])
            if num is not None:
                return num

        if left >= 0:
            num = clean_number(current_row.iloc[left])
            if num is not None:
                return num

    # 4) Nearby columns in next rows (closest first)
    for r in range(row_idx + 1, min(row_idx + 1 + max_row_lookahead, len(df))):
        next_row = df.iloc[r]
        for offset in range(1, max_col_offset + 1):
            right = col_idx + offset
            left = col_idx - offset

            if right < len(next_row):
                num = clean_number(next_row.iloc[right])
                if num is not None:
                    return num

            if left >= 0:
                num = clean_number(next_row.iloc[left])
                if num is not None:
                    return num

    return None


def find_section_value(df, section_keyword, property_keyword, search_depth=6):
    """
    Find section row (e.g., شربت خام)
    Then search next rows for property (e.g., رنگ)
    """

    for i in range(len(df)):

        row = df.iloc[i].astype(str)

        # Step 1: Find section
        if any(section_keyword in cell for cell in row):

            # Step 2: Search next few rows
            for j in range(i, min(i + search_depth, len(df))):
                subrow = df.iloc[j].astype(str)

                for col_idx, cell in enumerate(subrow):
                    if property_keyword in cell:
                        value = extract_number_near_property(df, j, col_idx)
                        if value is not None:
                            return value

                # Fallback: old behavior if property exists but
                # no position-based number can be extracted
                if any(property_keyword in cell for cell in subrow):
                    numbers = []
                    for cell in subrow:
                        num = clean_number(cell)
                        if num is not None:
                            numbers.append(num)

                    if numbers:
                        return numbers[0]

    return None


def find_day_night_columns(df):
    """
    Detect day/night column anchors from header rows containing "روز" and "شب".
    Returns (day_col_idx, night_col_idx). If not found, returns (None, None).
    """
    for i in range(len(df)):
        row = df.iloc[i].astype(str)
        day_cols = [idx for idx, cell in enumerate(row) if "روز" in cell]
        night_cols = [idx for idx, cell in enumerate(row) if "شب" in cell]
        if day_cols and night_cols:
            return day_cols[0], night_cols[0]
    return None, None


def extract_number_near_column(df, row_idx, target_col_idx, max_row_lookahead=0, max_col_offset=2):
    """
    Extract a numeric value close to a target column anchor for a specific row.
    """
    if target_col_idx is None:
        return None

    # Same row first, closest columns first
    row = df.iloc[row_idx]
    for offset in range(0, max_col_offset + 1):
        for col_idx in (target_col_idx + offset, target_col_idx - offset):
            if 0 <= col_idx < len(row):
                num = clean_number(row.iloc[col_idx])
                if num is not None:
                    return num

    # Look in next rows near same anchor
    for r in range(row_idx + 1, min(row_idx + 1 + max_row_lookahead, len(df))):
        next_row = df.iloc[r]
        for offset in range(0, max_col_offset + 1):
            for col_idx in (target_col_idx + offset, target_col_idx - offset):
                if 0 <= col_idx < len(next_row):
                    num = clean_number(next_row.iloc[col_idx])
                    if num is not None:
                        return num

    return None


def find_section_shift_values(
    df,
    section_keyword,
    property_keyword,
    day_col_idx,
    night_col_idx,
    search_depth=6,
    max_row_lookahead=0,
):
    """
    Find values for both day and night shifts for a section/property pair.
    Returns (day_value, night_value).
    """
    for i in range(len(df)):
        row = df.iloc[i].astype(str)

        if any(section_keyword in cell for cell in row):
            for j in range(i, min(i + search_depth, len(df))):
                subrow = df.iloc[j].astype(str)

                if any(property_keyword in cell for cell in subrow):
                    day_value = extract_number_near_column(
                        df, j, day_col_idx, max_row_lookahead=max_row_lookahead
                    )
                    night_value = extract_number_near_column(
                        df, j, night_col_idx, max_row_lookahead=max_row_lookahead
                    )

                    # Fallback if anchors fail: pick first two numeric values in row
                    if day_value is None or night_value is None:
                        nums = [clean_number(cell) for cell in df.iloc[j]]
                        nums = [n for n in nums if n is not None]
                        if day_value is None and len(nums) >= 1:
                            day_value = nums[0]
                        if night_value is None and len(nums) >= 2:
                            night_value = nums[1]

                    return day_value, night_value

    return None, None


def extract_value_near_column(df, row_idx, target_col_idx, max_row_lookahead=0, max_col_offset=2):
    """
    Extract a useful value close to a target column anchor for a specific row.
    Unlike extract_number_near_column, this preserves non-numeric values such as "10-12".
    """
    if target_col_idx is None:
        return None

    row = df.iloc[row_idx]
    for offset in range(0, max_col_offset + 1):
        for col_idx in (target_col_idx + offset, target_col_idx - offset):
            if 0 <= col_idx < len(row):
                value = clean_cell_value(row.iloc[col_idx])
                if value is not None:
                    return value

    for r in range(row_idx + 1, min(row_idx + 1 + max_row_lookahead, len(df))):
        next_row = df.iloc[r]
        for offset in range(0, max_col_offset + 1):
            for col_idx in (target_col_idx + offset, target_col_idx - offset):
                if 0 <= col_idx < len(next_row):
                    value = clean_cell_value(next_row.iloc[col_idx])
                    if value is not None:
                        return value

    return None


def find_section_shift_cell_values(
    df,
    section_keyword,
    property_keyword,
    day_col_idx,
    night_col_idx,
    search_depth=6,
    max_row_lookahead=0,
):
    """Find day/night values for a section/property pair while preserving text values."""
    for i in range(len(df)):
        row = df.iloc[i].astype(str)

        if any(section_keyword in cell for cell in row):
            for j in range(i, min(i + search_depth, len(df))):
                subrow = df.iloc[j].astype(str)

                if any(property_keyword in cell for cell in subrow):
                    day_value = extract_value_near_column(
                        df, j, day_col_idx, max_row_lookahead=max_row_lookahead
                    )
                    night_value = extract_value_near_column(
                        df, j, night_col_idx, max_row_lookahead=max_row_lookahead
                    )

                    if day_value is None or night_value is None:
                        values = [clean_cell_value(cell) for cell in df.iloc[j]]
                        values = [value for value in values if value is not None]
                        if day_value is None and len(values) >= 1:
                            day_value = values[0]
                        if night_value is None and len(values) >= 2:
                            night_value = values[1]

                    return day_value, night_value

    return None, None


def extract_first_number(x):
    """Extract the first numeric token from a text cell, supporting Persian digits."""
    if pd.isna(x):
        return None

    text = str(x)
    persian_digits = "۰۱۲۳۴۵۶۷۸۹"
    english_digits = "0123456789"
    for p, e in zip(persian_digits, english_digits):
        text = text.replace(p, e)
    text = text.replace("٫", ".")

    match = re.search(r"\d+(?:\.\d+)?", text)
    if not match:
        return None

    return clean_number(match.group(0))


def find_row_containing(df, keyword, start_row=0):
    """Return the first row index containing keyword, or None when missing."""
    for row_idx in range(start_row, len(df)):
        row = df.iloc[row_idx].astype(str)
        if any(keyword in cell for cell in row):
            return row_idx
    return None


def first_column_containing(df, row_idx, keyword):
    """Return the first column index in a row containing keyword, or None."""
    row = df.iloc[row_idx].astype(str)
    for col_idx, cell in enumerate(row):
        if keyword in cell:
            return col_idx
    return None


def extract_numeric_to_right(df, row_idx, col_idx, max_offset=4):
    """Extract the first numeric value to the right of a label cell."""
    row = df.iloc[row_idx]
    for offset in range(1, max_offset + 1):
        target_col = col_idx + offset
        if target_col >= len(row):
            break
        value = clean_number(row.iloc[target_col])
        if value is not None:
            return value
    return None


def extract_report_metadata(df):
    """Extract report header fields that apply to both day and night rows."""
    metadata = {}

    for row_idx in range(len(df)):
        row = df.iloc[row_idx].astype(str)
        for col_idx, cell in enumerate(row):
            if "سال" in cell and "report_year" not in metadata:
                metadata["report_year"] = extract_first_number(cell)
                if metadata["report_year"] is None:
                    metadata["report_year"] = extract_value_near_column(
                        df, row_idx, col_idx, max_col_offset=2
                    )

            if "دوره کاری" in cell and "work_period" not in metadata:
                metadata["work_period"] = extract_numeric_to_right(
                    df, row_idx, col_idx, max_offset=4
                )

            if "روز کاری" in cell and "work_day" not in metadata:
                metadata["work_day"] = extract_numeric_to_right(
                    df, row_idx, col_idx, max_offset=4
                )

            if "تاریخ" in cell and "report_date_day" not in metadata:
                metadata["report_date_day"] = extract_value_near_column(
                    df, row_idx, col_idx + 1, max_col_offset=1
                )
                metadata["report_date_month"] = extract_value_near_column(
                    df, row_idx, col_idx + 2, max_col_offset=1
                )
                metadata["report_date_year"] = extract_value_near_column(
                    df, row_idx, col_idx + 3, max_col_offset=1
                )

    return metadata


def normalize_table_label(value):
    """Normalize a row-label cell so R1/R2/R3/A/B/C matching is stable."""
    if pd.isna(value):
        return ""

    text = str(value).strip()
    persian_digits = "۰۱۲۳۴۵۶۷۸۹"
    english_digits = "0123456789"
    for p, e in zip(persian_digits, english_digits):
        text = text.replace(p, e)

    text = text.replace("ي", "ی").replace("ك", "ک")
    return re.sub(r"\s+", "", text).upper()


def normalize_metric_header(value):
    """Return the canonical metric name for a table header cell."""
    if pd.isna(value):
        return None

    text = str(value).strip()
    normalized = text.replace(" ", "").replace("_", "").upper()

    if normalized in {"BX", "BRIX"} or "بریکس" in text:
        return "brix"
    if normalized == "POL" or "پل" in text:
        return "pol"
    if normalized == "Q" or "خلوص" in text:
        return "q"
    if normalized in {"PH", "P.H"}:
        return "pH"

    return None


def row_contains_metric_headers(df, row_idx):
    """Return True when a row contains at least one BX/POL/Q/PH header."""
    return any(normalize_metric_header(cell) is not None for cell in df.iloc[row_idx])


def find_table_header_row(df, section_row, search_depth=10):
    """
    Locate the header row for matrix tables.

    The section title (for example پخت ها) may be on a separate row from the
    header row. Prefer a row below the section title that contains نوع plus the
    BX/POL/Q/PH headers. If a section such as پساب ها omits its own header, fall
    back to the nearest previous header row with the same structure.
    """
    forward_stop = min(section_row + search_depth + 1, len(df))
    for row_idx in range(section_row, forward_stop):
        row = df.iloc[row_idx].astype(str)
        if any("نوع" in cell for cell in row) and row_contains_metric_headers(df, row_idx):
            return row_idx

    for row_idx in range(section_row - 1, -1, -1):
        row = df.iloc[row_idx].astype(str)
        if any("نوع" in cell for cell in row) and row_contains_metric_headers(df, row_idx):
            return row_idx

    return None


def detected_metric_columns(df, header_row, label_col, required_metrics):
    """
    Detect day/night metric columns around the نوع column.

    Supports both observed layouts:
      * نوع | BX | POL | Q | PH | BX | POL | Q | PH
      * PH | Q | POL | BX | PH | Q | POL | BX | نوع
    """
    metric_headers = []
    for col_idx, cell in enumerate(df.iloc[header_row]):
        metric_name = normalize_metric_header(cell)
        if metric_name is not None:
            metric_headers.append((col_idx, metric_name))

    if not metric_headers:
        return {}, {}, metric_headers

    metric_headers = sorted(metric_headers)
    required_count = len(required_metrics)

    if label_col is not None:
        right_headers = [(col, metric) for col, metric in metric_headers if col > label_col]
        left_headers = [(col, metric) for col, metric in metric_headers if col < label_col]

        if len(right_headers) >= required_count * 2:
            ordered_headers = right_headers
        elif len(left_headers) >= required_count * 2:
            ordered_headers = left_headers
        else:
            ordered_headers = metric_headers
    else:
        ordered_headers = metric_headers

    first_group = ordered_headers[:required_count]
    second_group = ordered_headers[required_count : required_count * 2]

    # If there are extra metric-like cells, keep the two groups closest to the
    # label column. This protects against nearby summary rows/merged headers.
    if label_col is not None and len(ordered_headers) > required_count * 2:
        if ordered_headers[0][0] < label_col:
            first_group = ordered_headers[-required_count * 2 : -required_count]
            second_group = ordered_headers[-required_count:]
        else:
            first_group = ordered_headers[:required_count]
            second_group = ordered_headers[required_count : required_count * 2]

    return ({metric: col for col, metric in first_group},
            {metric: col for col, metric in second_group},
            metric_headers)


def extract_table_by_row_labels(
    df,
    section_keyword,
    row_labels,
    day_metrics,
    night_metrics=None,
    prefix="",
    max_rows=None,
):
    """
    Extract two-shift matrix sections such as پخت ها and پساب ها.

    The previous implementation assumed that the section title row was also the
    header row and that the row label appeared before the measurement columns.
    Real reports can put the section title on a separate row, put نوع at the far
    right, and order the measurement columns dynamically. This version finds the
    section, finds the actual نوع/BX/POL/Q/PH header row, detects both metric
    groups by header text, and then reads only rows whose labels match the table.
    """
    if night_metrics is None:
        night_metrics = day_metrics

    section_row = find_row_containing(df, section_keyword)
    debug_name = prefix or section_keyword
    print(f"[DEBUG] {debug_name}: section row = {section_row}")

    if section_row is None:
        return {}, {}

    header_row = find_table_header_row(df, section_row)
    print(f"[DEBUG] {debug_name}: header row = {header_row}")

    label_col = first_column_containing(df, header_row, "نوع") if header_row is not None else None

    if label_col is None:
        # Fallback for malformed sheets: infer the label column from the first
        # expected row label found inside this section.
        scan_rows = max_rows if max_rows is not None else len(row_labels) + 4
        normalized_labels = {normalize_table_label(label) for label in row_labels}
        for row_idx in range(section_row, min(section_row + scan_rows + 1, len(df))):
            for col_idx, cell in enumerate(df.iloc[row_idx]):
                if normalize_table_label(cell) in normalized_labels:
                    label_col = col_idx
                    break
            if label_col is not None:
                break

    day_columns, night_columns, all_metric_headers = ({}, {}, [])
    if header_row is not None:
        day_columns, night_columns, all_metric_headers = detected_metric_columns(
            df, header_row, label_col, day_metrics
        )

    print(f"[DEBUG] {debug_name}: label column = {label_col}")
    print(f"[DEBUG] {debug_name}: detected metric headers = {all_metric_headers}")
    print(f"[DEBUG] {debug_name}: day columns = {day_columns}")
    print(f"[DEBUG] {debug_name}: night columns = {night_columns}")

    normalized_row_labels = {
        normalize_table_label(label): normalized for label, normalized in row_labels.items()
    }

    day_values = {}
    night_values = {}
    for normalized_label in row_labels.values():
        for metric_name in day_metrics:
            day_values[f"{prefix}_{normalized_label}_{metric_name}"] = None
        for metric_name in night_metrics:
            night_values[f"{prefix}_{normalized_label}_{metric_name}"] = None

    if label_col is None or not day_columns or not night_columns:
        print(f"[DEBUG] {debug_name}: skipped extraction because columns could not be detected")
        return day_values, night_values

    if header_row is not None and header_row >= section_row:
        first_data_row = header_row + 1
    else:
        first_data_row = section_row

    rows_to_scan = max_rows if max_rows is not None else len(row_labels) + 4
    last_data_row = min(first_data_row + rows_to_scan, len(df))

    for row_idx in range(first_data_row, last_data_row):
        if label_col >= len(df.iloc[row_idx]):
            continue

        label_value = normalize_table_label(df.iloc[row_idx, label_col])
        if label_value not in normalized_row_labels:
            continue

        normalized_label = normalized_row_labels[label_value]
        extracted_day = {}
        extracted_night = {}

        for metric_name in day_metrics:
            metric_col = day_columns.get(metric_name)
            value = clean_cell_value(df.iloc[row_idx, metric_col]) if metric_col is not None else None
            day_values[f"{prefix}_{normalized_label}_{metric_name}"] = value
            extracted_day[metric_name] = value

        for metric_name in night_metrics:
            metric_col = night_columns.get(metric_name)
            value = clean_cell_value(df.iloc[row_idx, metric_col]) if metric_col is not None else None
            night_values[f"{prefix}_{normalized_label}_{metric_name}"] = value
            extracted_night[metric_name] = value

        print(
            f"[DEBUG] {debug_name}: row {row_idx}, label {normalized_label}, "
            f"day = {extracted_day}, night = {extracted_night}"
        )

    return day_values, night_values

def extract_white_sugar_quality_shift_values(df):
    """Extract the per-shift white sugar quality/point rows."""
    section_row = find_row_containing(df, "کیفیت شکر سفید")
    if section_row is None:
        return {}, {}

    label_col = first_column_containing(df, section_row, "رطوبت")
    if label_col is None:
        label_col = 2

    day_start_col = label_col + 1
    day_sample_cols = 3
    night_start_col = day_start_col + day_sample_cols + 1
    night_sample_cols = 4
    row_map = {
        "رطوبت": "moisture",
        "رنگ محلول": "solution_color",
        "رنگ ظاهری": "apparent_color",
        "خاکستر": "ash",
        "جمع پوئن": "total_points",
    }
    day_values = {}
    night_values = {}

    for row_idx in range(section_row, min(section_row + len(row_map) + 2, len(df))):
        label_text = str(df.iloc[row_idx, label_col]).strip()
        metric_name = None
        for keyword, normalized in row_map.items():
            if keyword in label_text:
                metric_name = normalized
                break
        if metric_name is None:
            continue

        for sample_offset in range(day_sample_cols):
            value = clean_cell_value(df.iloc[row_idx, day_start_col + sample_offset])
            day_values[f"white_quality_{metric_name}_{sample_offset + 1}"] = value

        for sample_offset in range(night_sample_cols):
            value = clean_cell_value(df.iloc[row_idx, night_start_col + sample_offset])
            night_values[f"white_quality_{metric_name}_{sample_offset + 1}"] = value

    return day_values, night_values


def extract_boiler_shift_values(df):
    """Extract boiler water measurements for both shifts."""
    header_row = find_row_containing(df, "مشخصات آب مصرفی")
    if header_row is None or header_row + 1 >= len(df):
        return {}, {}

    metrics = ["pH", "alkalinity", "hardness", "tds"]
    first_metric_col = first_column_containing(df, header_row, "pH")
    if first_metric_col is None:
        first_metric_col = 3

    day_values = {}
    night_values = {}
    for offset, metric_name in enumerate(metrics):
        day_values[f"boiler_{metric_name}"] = clean_cell_value(
            df.iloc[header_row + 1, first_metric_col + offset]
        )
        night_values[f"boiler_{metric_name}"] = clean_cell_value(
            df.iloc[header_row + 1, first_metric_col + len(metrics) + offset]
        )

    return day_values, night_values


def extract_average_row_values(df, section_keyword, prefix):
    """Extract average-summary rows whose headers are followed by values in the next row."""
    header_row = find_row_containing(df, section_keyword)
    if header_row is None or header_row + 1 >= len(df):
        return {}

    metric_map = {
        "رطوبت": "moisture",
        "اینورت": "invert",
        "رنگ محلول": "solution_color",
        "رنگ ظاهری": "apparent_color",
        "خاکستر": "ash",
        "جمع پوئن": "total_points",
    }
    values = {}
    for col_idx, cell in enumerate(df.iloc[header_row].astype(str)):
        for keyword, metric_name in metric_map.items():
            if keyword in cell:
                values[f"{prefix}_{metric_name}"] = clean_cell_value(
                    df.iloc[header_row + 1, col_idx]
                )

        if "سولفیت" in cell:
            values[f"{prefix}_sulfite"] = clean_cell_value(cell)

    for cell in df.iloc[header_row + 1].astype(str):
        if "MA" in cell or "CV" in cell:
            values[f"{prefix}_ma_cv"] = clean_cell_value(cell)

    return values


# ==========================================
# 4️⃣ EXCEL PROCESSING
# ==========================================


def extract_excel_to_dataframe(input_file):
    """Extract all worksheets from the Excel report into a cleaned dataframe."""
    xls = pd.ExcelFile(input_file)
    all_sheets = xls.sheet_names

    all_days_data = []

    for sheet in all_sheets:

        print(f"Processing sheet: {sheet}")

        df = pd.read_excel(input_file, sheet_name=sheet, header=None)
        df = df.fillna("").astype(str)

        day_col_idx, night_col_idx = find_day_night_columns(df)

        day_data = {"sheet_name": sheet, "shift_name": "0"}
        night_data = {"sheet_name": sheet, "shift_name": "1"}

        report_metadata = extract_report_metadata(df)
        day_data.update(report_metadata)
        night_data.update(report_metadata)

        # -------------------------------
        # RAW SUGAR
        # -------------------------------
        day_value, night_value = find_section_shift_values(df, "شکرخام", "رنگ", day_col_idx, night_col_idx)
        day_data["raw_sugar_color"] = day_value
        night_data["raw_sugar_color"] = night_value

        # -------------------------------
        # RAW SYRUP
        # -------------------------------
        day_value, night_value = find_section_shift_values(df, "شربت خام", "بریکس", day_col_idx, night_col_idx)
        day_data["raw_syrup_brix"] = day_value
        night_data["raw_syrup_brix"] = night_value

        day_value, night_value = find_section_shift_values(df, "شربت خام", "رنگ", day_col_idx, night_col_idx)
        day_data["raw_syrup_color"] = day_value
        night_data["raw_syrup_color"] = night_value

        # -------------------------------
        # LIME MILK + LIME TREATED SYRUP
        # -------------------------------
        day_value, night_value = find_section_shift_cell_values(df, "بومه", "بومه", day_col_idx, night_col_idx)
        day_data["lime_milk_baume"] = day_value
        night_data["lime_milk_baume"] = night_value

        day_value, night_value = find_section_shift_values(df, "شربت آهک", "قلیایی", day_col_idx, night_col_idx)
        day_data["lime_alkalinity"] = day_value
        night_data["lime_alkalinity"] = night_value

        day_value, night_value = find_section_shift_values(df, "CO2", "درصد", day_col_idx, night_col_idx)
        day_data["co2_percent"] = day_value
        night_data["co2_percent"] = night_value

        # -------------------------------
        # CARBONATED SYRUP
        # -------------------------------
        day_value, night_value = find_section_shift_values(df, "شربت کربناتور", "قلیایی", day_col_idx, night_col_idx)
        day_data["carbonated_alkalinity"] = day_value
        night_data["carbonated_alkalinity"] = night_value

        day_value, night_value = find_section_shift_values(df, "شربت کربناتور", "PH", day_col_idx, night_col_idx)
        day_data["carbonated_pH"] = day_value
        night_data["carbonated_pH"] = night_value

        # -------------------------------
        # FILTER CAKE
        # -------------------------------
        day_value, night_value = find_section_shift_values(df, "گل فیلتر", "رطوبت", day_col_idx, night_col_idx)
        day_data["filtercake_moisture"] = day_value
        night_data["filtercake_moisture"] = night_value

        day_value, night_value = find_section_shift_values(df, "گل فیلتر", "قند", day_col_idx, night_col_idx)
        day_data["filtercake_sugar"] = day_value
        night_data["filtercake_sugar"] = night_value

        average_filtercake_sugar = find_section_value(df, "میانگین قند گل فیلتر", "میانگین قند گل فیلتر")
        day_data["filtercake_sugar_average_to_date"] = average_filtercake_sugar
        night_data["filtercake_sugar_average_to_date"] = average_filtercake_sugar

        # -------------------------------
        # SWEET WATER
        # -------------------------------
        day_value, night_value = find_section_shift_values(df, "آب شیرین", "بریکس", day_col_idx, night_col_idx)
        day_data["sweetwater_brix"] = day_value
        night_data["sweetwater_brix"] = night_value

        # -------------------------------
        # SULPHITED SYRUP
        # -------------------------------
        day_value, night_value = find_section_shift_values(df, "شربت سولفیته", "PH", day_col_idx, night_col_idx)
        day_data["sulphited_pH"] = day_value
        night_data["sulphited_pH"] = night_value

        day_value, night_value = find_section_shift_values(df, "شربت سولفیته", "بریکس", day_col_idx, night_col_idx)
        day_data["sulphited_brix"] = day_value
        night_data["sulphited_brix"] = night_value

        day_value, night_value = find_section_shift_values(df, "شربت سولفیته", "رنگ", day_col_idx, night_col_idx)
        day_data["sulphited_color"] = day_value
        night_data["sulphited_color"] = night_value

        # -------------------------------
        # STANDARD LIQUOR
        # -------------------------------
        day_value, night_value = find_section_shift_values(df, "لیکور استاندارد", "PH", day_col_idx, night_col_idx)
        day_data["standard_liquor_pH"] = day_value
        night_data["standard_liquor_pH"] = night_value

        day_value, night_value = find_section_shift_values(df, "لیکور استاندارد", "بریکس", day_col_idx, night_col_idx)
        day_data["standard_liquor_brix"] = day_value
        night_data["standard_liquor_brix"] = night_value

        day_value, night_value = find_section_shift_values(df, "لیکور استاندارد", "رنگ", day_col_idx, night_col_idx)
        day_data["standard_liquor_color"] = day_value
        night_data["standard_liquor_color"] = night_value

        # -------------------------------
        # BOILING MASSECUITES (پخت ها)
        # -------------------------------
        row_labels = {"R1": "r1", "R2": "r2", "R3": "r3", "A": "a", "B": "b", "C": "c"}
        day_values, night_values = extract_table_by_row_labels(
            df,
            "پخت",
            row_labels,
            ["brix", "pol", "q", "pH"],
            prefix="boiling",
            max_rows=8,
        )
        day_data.update(day_values)
        night_data.update(night_values)

        # -------------------------------
        # WASTEWATER / RUNOFFS (پساب ها)
        # -------------------------------
        wastewater_row_labels = {
            "R1": "r1",
            "R2": "r2",
            "R3": "r3",
            "A": "a",
            "B": "b",
            "C": "c",
            "ملاس": "molasses",
        }
        day_values, night_values = extract_table_by_row_labels(
            df,
            "پساب",
            wastewater_row_labels,
            ["brix", "pol", "q", "pH"],
            prefix="wastewater",
            max_rows=9,
        )
        day_data.update(day_values)
        night_data.update(night_values)

        # -------------------------------
        # WHITE SUGAR QUALITY SAMPLES + POINTS
        # -------------------------------
        day_values, night_values = extract_white_sugar_quality_shift_values(df)
        day_data.update(day_values)
        night_data.update(night_values)

        # -------------------------------
        # FINAL WHITE SUGAR (AVERAGE TWO SHIFTS)
        # -------------------------------
        white_moisture = find_section_value(df, "میانگین نتایج شکر سفید در دو شیفت", "رطوبت")
        white_invert = find_section_value(df, "میانگین نتایج شکر سفید در دو شیفت", "اینورت")
        white_solution_color = find_section_value(df, "میانگین نتایج شکر سفید در دو شیفت", "رنگ محلول")
        white_apparent_color = find_section_value(df, "میانگین نتایج شکر سفید در دو شیفت", "رنگ ظاهری")
        white_ash = find_section_value(df, "میانگین نتایج شکر سفید در دو شیفت", "خاکستر")
        white_total_points = find_section_value(df, "میانگین نتایج شکر سفید در دو شیفت", "جمع پوئن")

        # These are already two-shift averages, so keep same value for both rows.
        day_data["white_moisture"] = white_moisture
        day_data["white_invert"] = white_invert
        day_data["white_solution_color"] = white_solution_color
        day_data["white_apparent_color"] = white_apparent_color
        day_data["white_ash"] = white_ash
        day_data["white_total_points"] = white_total_points

        night_data["white_moisture"] = white_moisture
        night_data["white_invert"] = white_invert
        night_data["white_solution_color"] = white_solution_color
        night_data["white_apparent_color"] = white_apparent_color
        night_data["white_ash"] = white_ash
        night_data["white_total_points"] = white_total_points

        white_average_values = extract_average_row_values(
            df, "میانگین نتایج شکر سفید در دو شیفت", "white_average_two_shifts"
        )
        day_data.update(white_average_values)
        night_data.update(white_average_values)

        white_to_date_values = extract_average_row_values(
            df, "میانگین نتایج شکر سفید تا این تاریخ", "white_average_to_date"
        )
        day_data.update(white_to_date_values)
        night_data.update(white_to_date_values)

        # -------------------------------
        # BOILER WATER
        # -------------------------------
        day_values, night_values = extract_boiler_shift_values(df)
        day_data.update(day_values)
        night_data.update(night_values)

        all_days_data.append(day_data)
        all_days_data.append(night_data)

    return pd.DataFrame(all_days_data)


# ==========================================
# 5️⃣ CSV SAVING + MERGING
# ==========================================


def remove_duplicate_rows(df):
    """Remove duplicate rows and print how many rows were dropped."""
    before_count = len(df)
    df = df.drop_duplicates()
    removed_count = before_count - len(df)
    print(f"Removed {removed_count} duplicate row(s).")
    return df


def save_csv(df, output_file):
    """Save a dataframe as CSV."""
    df.to_csv(output_file, index=False)
    print(f"Output file: {output_file}")


def merge_csv_files(csv_files, output_file, drop_duplicates=False):
    """Merge multiple CSV files into one output CSV."""
    frames = []
    for csv_file in csv_files:
        print(f"Reading CSV: {csv_file}")
        frames.append(pd.read_csv(csv_file))

    merged_df = pd.concat(frames, ignore_index=True, sort=False)

    if drop_duplicates:
        merged_df = remove_duplicate_rows(merged_df)

    save_csv(merged_df, output_file)
    return merged_df


def run_excel_conversion():
    """Prompt for Excel conversion inputs and write the extracted CSV."""
    input_file = ask_text("Input Excel file", DEFAULT_INPUT_FILE)
    output_file = ask_text("Output CSV file", DEFAULT_OUTPUT_FILE)

    input_path = resolve_existing_path(input_file)
    if not input_path.exists():
        raise FileNotFoundError(f"Input Excel file was not found: {input_file}")

    final_df = extract_excel_to_dataframe(input_path)

    if ask_yes_no("Remove duplicated rows from this output", default=False):
        final_df = remove_duplicate_rows(final_df)

    save_csv(final_df, output_file)
    print("\n✅ EXTRACTION COMPLETED SUCCESSFULLY")

    return Path(output_file)


def run_csv_merge(default_files=None):
    """Prompt for CSV merge inputs and write the merged CSV."""
    if default_files:
        default_file_text = ", ".join(str(file_path) for file_path in default_files)
        csv_file_text = ask_text("CSV files to merge, separated by commas", default_file_text)
        csv_files = [resolve_existing_path(item.strip()) for item in csv_file_text.split(",") if item.strip()]
        missing_files = [str(file_path) for file_path in csv_files if not file_path.exists()]
        if missing_files:
            raise FileNotFoundError("CSV file(s) not found: " + ", ".join(missing_files))
    else:
        csv_files = ask_csv_files("CSV files to merge, separated by commas")

    output_file = ask_text("Merged output CSV file", "merged_output.csv")
    drop_duplicates = ask_yes_no("Remove duplicated rows from merged CSV", default=True)
    merge_csv_files(csv_files, output_file, drop_duplicates=drop_duplicates)
    print("\n✅ MERGE COMPLETED SUCCESSFULLY")

    return Path(output_file)


def main():
    print("Sugar report converter")
    print("1) Convert Excel to CSV")
    print("2) Merge CSV files")
    print("3) Convert Excel to CSV, then optionally merge CSV files")

    while True:
        choice = ask_text("Choose an option", "1")
        if choice in {"1", "2", "3"}:
            break
        print("Please choose 1, 2, or 3.")

    generated_csv = None
    if choice in {"1", "3"}:
        generated_csv = run_excel_conversion()

    if choice == "2":
        run_csv_merge()
    elif choice == "3" and ask_yes_no("Merge CSV files now", default=False):
        default_files = [generated_csv] if generated_csv else None
        run_csv_merge(default_files=default_files)


if __name__ == "__main__":
    main()
