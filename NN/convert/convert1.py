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
        # LIME TREATED SYRUP
        # -------------------------------
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
