"""Interactive data-cleaning helper for CSV and Excel files.

Run from this folder with:
    python clean.py

The script asks for input/output filenames and then presents a menu for common
cleanup tasks such as removing duplicates, extracting numbers from cells like
"x=12.5", dropping sparse rows/columns, and keeping only complete rows.
"""

import re
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_FILE = "input.csv"
DEFAULT_OUTPUT_FILE = "cleaned.csv"
EMPTY_MARKERS = {"", "nan", "none", "null", "na", "n/a", "...", "…", "-"}
ASSIGNMENT_NUMBER_RE = re.compile(
    r"^\s*[^=:\s][^=:]*\s*[=:]\s*([-+]?\d[\d,]*(?:[\.٫]\d+)?(?:[eE][-+]?\d+)?)\s*$"
)
ANY_NUMBER_RE = re.compile(r"[-+]?\d[\d,]*(?:[\.٫]\d+)?(?:[eE][-+]?\d+)?")


# ==========================================
# 1) Interactive helpers
# ==========================================


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


def ask_int(prompt, default=None, minimum=None, maximum=None):
    """Ask for an integer with optional default and bounds."""
    while True:
        raw_value = ask_text(prompt, default)
        try:
            value = int(raw_value)
        except (TypeError, ValueError):
            print("Please enter a whole number.")
            continue

        if minimum is not None and value < minimum:
            print(f"Please enter a number greater than or equal to {minimum}.")
            continue
        if maximum is not None and value > maximum:
            print(f"Please enter a number less than or equal to {maximum}.")
            continue
        return value


def ask_float(prompt, default=None, minimum=None, maximum=None):
    """Ask for a floating point value with optional default and bounds."""
    while True:
        raw_value = ask_text(prompt, default)
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            print("Please enter a number.")
            continue

        if minimum is not None and value < minimum:
            print(f"Please enter a number greater than or equal to {minimum}.")
            continue
        if maximum is not None and value > maximum:
            print(f"Please enter a number less than or equal to {maximum}.")
            continue
        return value


def resolve_path(file_path):
    """Return a path from the current directory or this script directory."""
    path = Path(file_path).expanduser()
    if path.is_absolute() or path.exists():
        return path

    script_relative_path = SCRIPT_DIR / path
    if script_relative_path.exists():
        return script_relative_path

    return path


def ask_existing_file(prompt, default=None):
    """Ask for an existing input file."""
    while True:
        file_path = resolve_path(ask_text(prompt, default))
        if file_path.exists():
            return file_path
        print(f"File not found: {file_path}")


def ask_output_file(prompt, default=None):
    """Ask for an output file, defaulting to this script folder for relative paths."""
    raw_path = ask_text(prompt, default)
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path
    return SCRIPT_DIR / path


# ==========================================
# 2) Data loading/saving helpers
# ==========================================


def load_dataframe(input_file):
    """Load a CSV or Excel file into a DataFrame."""
    suffix = input_file.suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(input_file)

    if suffix in {".xlsx", ".xls"}:
        sheet_name = ask_text("Excel sheet name or number (press Enter for first sheet)", "0")
        try:
            sheet = int(sheet_name)
        except ValueError:
            sheet = sheet_name
        return pd.read_excel(input_file, sheet_name=sheet)

    raise ValueError("Unsupported input file. Please use .csv, .xlsx, or .xls.")


def save_dataframe(df, output_file):
    """Save a DataFrame as CSV or Excel based on the output extension."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_file.suffix.lower()

    if suffix in {".xlsx", ".xls"}:
        df.to_excel(output_file, index=False)
        return

    if suffix != ".csv":
        output_file = output_file.with_suffix(".csv")
    df.to_csv(output_file, index=False)


def normalize_empty_values(df):
    """Convert blank-looking string cells to pandas NA so row filters work."""
    cleaned = df.copy()
    for column in cleaned.columns:
        if cleaned[column].dtype == "object":
            text_values = cleaned[column].astype(str).str.strip().str.lower()
            cleaned.loc[text_values.isin(EMPTY_MARKERS), column] = pd.NA
    return cleaned


def show_stats(df, title="Current data"):
    """Print a small summary of the current data."""
    total_cells = df.shape[0] * df.shape[1]
    empty_cells = int(df.isna().sum().sum())
    duplicate_rows = int(df.duplicated().sum())
    empty_percent = (empty_cells / total_cells * 100) if total_cells else 0

    print(f"\n--- {title} ---")
    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")
    print(f"Duplicate rows: {duplicate_rows}")
    print(f"Empty cells: {empty_cells} ({empty_percent:.1f}%)")


# ==========================================
# 3) Cleaning helpers
# ==========================================


def normalize_number_text(value):
    """Normalize Persian digits, decimal marks, and thousands separators."""
    text = str(value).strip()

    persian_digits = "۰۱۲۳۴۵۶۷۸۹"
    arabic_digits = "٠١٢٣٤٥٦٧٨٩"
    english_digits = "0123456789"
    for source_digits in (persian_digits, arabic_digits):
        for source, english in zip(source_digits, english_digits):
            text = text.replace(source, english)

    return text.replace("٫", ".").replace(",", "")


def extract_number(value, require_assignment=False):
    """Extract a numeric value from text, optionally requiring x=number style text."""
    if pd.isna(value):
        return pd.NA

    text = normalize_number_text(value)
    assignment_match = ASSIGNMENT_NUMBER_RE.search(text)
    if assignment_match:
        return float(assignment_match.group(1))

    if require_assignment:
        return value

    number_match = ANY_NUMBER_RE.search(text)
    if not number_match:
        return value

    return float(number_match.group(0))


def detect_problematic_columns(df):
    """Find columns containing cells such as x=12 or value: 7.5."""
    detected_columns = []
    for column in df.columns:
        series = df[column].dropna().astype(str).map(normalize_number_text)
        if series.map(lambda text: bool(ASSIGNMENT_NUMBER_RE.search(text))).any():
            detected_columns.append(column)
    return detected_columns


def ask_columns(df, detected_columns=None):
    """Ask which columns should be cleaned or dropped."""
    detected_columns = detected_columns or []
    print("\nAvailable columns:")
    for index, column in enumerate(df.columns, start=1):
        marker = "  *detected*" if column in detected_columns else ""
        print(f"  {index}. {column}{marker}")

    print("\nEnter column numbers/names separated by commas.")
    print("Use 'all' for all columns or 'detected' for detected problematic columns.")

    while True:
        raw_value = ask_text("Columns", "detected" if detected_columns else "all")
        lowered = raw_value.strip().lower()

        if lowered == "all":
            return list(df.columns)
        if lowered == "detected":
            if detected_columns:
                return detected_columns
            print("No detected problematic columns. Choose specific columns or 'all'.")
            continue

        selected_columns = []
        for part in [item.strip() for item in raw_value.split(",") if item.strip()]:
            if part.isdigit():
                index = int(part) - 1
                if 0 <= index < len(df.columns):
                    selected_columns.append(df.columns[index])
                else:
                    print(f"Column number out of range: {part}")
                    break
            elif part in df.columns:
                selected_columns.append(part)
            else:
                print(f"Unknown column: {part}")
                break
        else:
            if selected_columns:
                return list(dict.fromkeys(selected_columns))
            print("Please choose at least one column.")


def remove_duplicate_rows(df):
    """Remove exact duplicate rows."""
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"Removed {before - len(df)} duplicate rows.")
    return df


def clean_problematic_columns(df):
    """Extract numbers from x=number cells or drop problematic columns."""
    detected_columns = detect_problematic_columns(df)
    if detected_columns:
        print("\nDetected columns with x=number or name:number cells:")
        for column in detected_columns:
            print(f"  - {column}")
    else:
        print("\nNo x=number style columns were detected automatically.")

    columns = ask_columns(df, detected_columns)
    print("\nChoose how to fix the selected columns:")
    print("  1. Convert only x=number / label:number cells to numbers")
    print("  2. Extract the first number from every non-empty cell")
    print("  3. Remove/drop the selected columns")
    action = ask_int("Action", "1", minimum=1, maximum=3)

    cleaned = df.copy()
    if action == 3:
        cleaned = cleaned.drop(columns=columns)
        print(f"Dropped {len(columns)} columns.")
        return cleaned

    require_assignment = action == 1
    for column in columns:
        cleaned[column] = cleaned[column].map(
            lambda value: extract_number(value, require_assignment=require_assignment)
        )
        numeric_version = pd.to_numeric(cleaned[column], errors="coerce")
        if numeric_version.notna().sum() == cleaned[column].notna().sum():
            cleaned[column] = numeric_version

    print(f"Cleaned {len(columns)} columns.")
    return cleaned


def remove_sparse_rows(df):
    """Remove rows that contain too many empty columns."""
    total_columns = len(df.columns)
    print("\nSparse row removal options:")
    print("  1. Require a minimum percent of filled columns")
    print("  2. Allow only a maximum number of empty columns")
    action = ask_int("Option", "1", minimum=1, maximum=2)

    before = len(df)
    if action == 1:
        min_percent = ask_float("Minimum filled percent", "50", minimum=0, maximum=100)
        min_filled = int((min_percent / 100) * total_columns)
        if min_percent > 0 and min_filled == 0:
            min_filled = 1
    else:
        max_empty = ask_int("Maximum empty columns allowed", "2", minimum=0, maximum=total_columns)
        min_filled = total_columns - max_empty

    cleaned = df.dropna(thresh=min_filled).reset_index(drop=True)
    print(f"Removed {before - len(cleaned)} sparse rows.")
    return cleaned


def remove_sparse_columns(df):
    """Optionally remove columns that are mostly empty."""
    min_percent = ask_float("Minimum filled percent for a column", "50", minimum=0, maximum=100)
    min_filled = int((min_percent / 100) * len(df))
    if min_percent > 0 and min_filled == 0:
        min_filled = 1

    before = len(df.columns)
    cleaned = df.dropna(axis=1, thresh=min_filled)
    print(f"Removed {before - len(cleaned.columns)} sparse columns.")
    return cleaned


def keep_complete_rows(df):
    """Keep only rows where every column has data."""
    before = len(df)
    cleaned = df.dropna(how="any").reset_index(drop=True)
    print(f"Removed {before - len(cleaned)} rows with missing data.")
    return cleaned


# ==========================================
# 4) Interactive menu
# ==========================================


def print_menu():
    """Print the available cleaning actions."""
    print("\nChoose a cleaning option:")
    print("  1. Remove duplicate rows")
    print("  2. Fix or remove problematic columns (x=number -> number)")
    print("  3. Remove rows with mostly empty values / many empty columns")
    print("  4. Keep only rows that have full data")
    print("  5. Remove columns that are mostly empty")
    print("  6. Show current data summary")
    print("  7. Save and exit")
    print("  8. Exit without saving")


def main():
    """Run the interactive cleaner."""
    print("Interactive data cleaner")
    print("Supported input files: .csv, .xlsx, .xls")

    input_file = ask_existing_file("Input filename", DEFAULT_INPUT_FILE)
    output_file = ask_output_file("Output filename", DEFAULT_OUTPUT_FILE)

    df = load_dataframe(input_file)
    df = normalize_empty_values(df)
    show_stats(df, "Loaded data")

    while True:
        print_menu()
        choice = ask_int("Option", minimum=1, maximum=8)

        if choice == 1:
            df = remove_duplicate_rows(df)
        elif choice == 2:
            df = clean_problematic_columns(df)
            df = normalize_empty_values(df)
        elif choice == 3:
            df = remove_sparse_rows(df)
        elif choice == 4:
            df = keep_complete_rows(df)
        elif choice == 5:
            df = remove_sparse_columns(df)
        elif choice == 6:
            show_stats(df)
        elif choice == 7:
            save_dataframe(df, output_file)
            print(f"Saved cleaned data to: {output_file}")
            break
        elif choice == 8:
            if ask_yes_no("Exit without saving", default=False):
                print("No file was saved.")
                break


if __name__ == "__main__":
    main()
