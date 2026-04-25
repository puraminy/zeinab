import pandas as pd

# ==========================================
# 1️⃣ SETTINGS
# ==========================================

input_file = "gozaresh.xlsx"
output_file = "sugar_all_days_clean_7.csv"

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
    except:
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
    max_row_lookahead=0
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


# ==========================================
# 5️⃣ SAVE FINAL CSV
# ==========================================

final_df = pd.DataFrame(all_days_data)
final_df.to_csv(output_file, index=False)

print("\n✅ EXTRACTION COMPLETED SUCCESSFULLY")
print("Output file:", output_file)
