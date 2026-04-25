import pandas as pd

# ==========================================
# 1️⃣ SETTINGS
# ==========================================

input_file = "gozaresh.xlsx"
output_file = "sugar_all_days_clean_4.csv"

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

                if any(property_keyword in cell for cell in subrow):

                    numbers = []
                    for cell in subrow:
                        num = clean_number(cell)
                        if num is not None:
                            numbers.append(num)

                    if numbers:
                        return numbers[0]

    return None


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

    day_data = {"sheet_name": sheet}

    # -------------------------------
    # RAW SUGAR
    # -------------------------------
    day_data["raw_sugar_color"] = find_section_value(df, "شکرخام", "رنگ")

    # -------------------------------
    # RAW SYRUP
    # -------------------------------
    day_data["raw_syrup_brix"] = find_section_value(df, "شربت خام", "بریکس")
    day_data["raw_syrup_color"] = find_section_value(df, "شربت خام", "رنگ")

    # -------------------------------
    # LIME TREATED SYRUP
    # -------------------------------
    day_data["lime_alkalinity"] = find_section_value(df, "شربت آهک", "قلیایی")
    day_data["co2_percent"] = find_section_value(df, "CO2", "درصد")

    # -------------------------------
    # CARBONATED SYRUP
    # -------------------------------
    day_data["carbonated_alkalinity"] = find_section_value(df, "شربت کربناتور", "قلیایی")
    day_data["carbonated_pH"] = find_section_value(df, "شربت کربناتور", "PH")

    # -------------------------------
    # FILTER CAKE
    # -------------------------------
    day_data["filtercake_moisture"] = find_section_value(df, "گل فیلتر", "رطوبت")
    day_data["filtercake_sugar"] = find_section_value(df, "گل فیلتر", "قند")

    # -------------------------------
    # SWEET WATER
    # -------------------------------
    day_data["sweetwater_brix"] = find_section_value(df, "آب شیرین", "بریکس")

    # -------------------------------
    # SULPHITED SYRUP
    # -------------------------------
    day_data["sulphited_pH"] = find_section_value(df, "شربت سولفیته", "PH")
    day_data["sulphited_brix"] = find_section_value(df, "شربت سولفیته", "بریکس")
    day_data["sulphited_color"] = find_section_value(df, "شربت سولفیته", "رنگ")

    # -------------------------------
    # STANDARD LIQUOR
    # -------------------------------
    day_data["standard_liquor_pH"] = find_section_value(df, "لیکور استاندارد", "PH")
    day_data["standard_liquor_brix"] = find_section_value(df, "لیکور استاندارد", "بریکس")
    day_data["standard_liquor_color"] = find_section_value(df, "لیکور استاندارد", "رنگ")

    # -------------------------------
    # FINAL WHITE SUGAR (AVERAGE TWO SHIFTS)
    # -------------------------------
    day_data["white_moisture"] = find_section_value(df, "میانگین نتایج شکر سفید در دو شیفت", "رطوبت")
    day_data["white_invert"] = find_section_value(df, "میانگین نتایج شکر سفید در دو شیفت", "اینورت")
    day_data["white_solution_color"] = find_section_value(df, "میانگین نتایج شکر سفید در دو شیفت", "رنگ محلول")
    day_data["white_apparent_color"] = find_section_value(df, "میانگین نتایج شکر سفید در دو شیفت", "رنگ ظاهری")
    day_data["white_ash"] = find_section_value(df, "میانگین نتایج شکر سفید در دو شیفت", "خاکستر")
    day_data["white_total_points"] = find_section_value(df, "میانگین نتایج شکر سفید در دو شیفت", "جمع پوئن")

    all_days_data.append(day_data)


# ==========================================
# 5️⃣ SAVE FINAL CSV
# ==========================================

final_df = pd.DataFrame(all_days_data)
final_df.to_csv(output_file, index=False)

print("\n✅ EXTRACTION COMPLETED SUCCESSFULLY")
print("Output file:", output_file)

