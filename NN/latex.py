import pandas as pd
from tabulate import tabulate

def generate_latex_table(table, caption="table", label=""):
    # Function to escape LaTeX special characters
    def escape_latex_special_chars(text):
        return (text.replace('_', '\\_')
                    .replace('%', '\\%')
                    .replace('$', '\\$')
                    .replace('&', '\\&')
                    .replace('#', '\\#')
                    .replace('{', '\\{')
                    .replace('}', '\\}'))

    # Add the new column "R2 mean ± std" if it does not exist
    if "R2" in table.columns and "R2 std" in table.columns:
        table["R2 mean ± std"] = table.apply(lambda row: f"{row['R2']} ± {row['R2 std']}", axis=1)

    # Apply the escaping function to all elements in the DataFrame
    escaped_table = table.applymap(lambda x: escape_latex_special_chars(str(x)) if isinstance(x, str) else x)

    # Process the first column for multirow
    if "model" in escaped_table:
        model_col = escaped_table['model'].tolist()
        multirow_model_col = []
        previous_model = None
        row_span = 0
        latex_table_lines = []

        for i, row in escaped_table.iterrows():
            model = row['model']
            if model == previous_model:
                multirow_model_col.append('')
                row_span += 1
            else:
                if row_span > 0:
                    latex_table_lines[-(row_span+1)][0] = f"\\multirow{{{row_span+1}}}{{*}}{{{previous_model}}}"
                    latex_table_lines.append([''] * len(row))  # placeholder for \hline
                row_span = 0
                multirow_model_col.append(model)
            previous_model = model
            latex_table_lines.append(row.tolist())

        if row_span > 0:
            latex_table_lines[-(row_span+1)][0] = f"\\multirow{{{row_span+1}}}{{*}}{{{previous_model}}}"
            latex_table_lines.append([''] * len(row))  # placeholder for \hline

    else:
        latex_table_lines = escaped_table.values.tolist()

    # Convert to LaTeX table using tabulate
    latex_table = tabulate(latex_table_lines, headers=escaped_table.columns, tablefmt='latex_raw', showindex=False)

    # Insert \hline at the beginning and after each group of model names
    latex_table_lines = latex_table.splitlines()
    latex_table_lines.insert(1, '\\hline')
    for i in range(len(latex_table_lines) - 2, 1, -1):
        if not latex_table_lines[i].startswith('\\multirow') and latex_table_lines[i+1].startswith('\\multirow'):
            latex_table_lines.insert(i+1, '\\hline')

    latex_table = '\n'.join(latex_table_lines)

    # Wrap the table in a table environment
    table_env = f"""
    \\begin{{table*}}
        \\centering
        {latex_table}
        \\caption{{{caption}}}
        \\label{{{label}}}
    \\end{{table*}}
    """

    return table_env

# Example usage with a sample DataFrame
results = [
    {"model": "model1", "R2": 76.09, "MSE": 1.23, "R2 std": 2.3, "R2 List": [75.0, 77.1], "hidden sizes": [64, 32], "total hs": 96, "epochs": 100},
    {"model": "model1", "R2": 74.50, "MSE": 1.20, "R2 std": 2.1, "R2 List": [74.0, 75.0], "hidden sizes": [64, 32], "total hs": 96, "epochs": 100},
    {"model": "model2", "R2": 43.37, "MSE": 2.34, "R2 std": 3.4, "R2 List": [42.0, 44.7], "hidden sizes": [128, 64], "total hs": 192, "epochs": 150},
    {"model": "model3", "R2": 88.38, "MSE": 0.98, "R2 std": 1.8, "R2 List": [87.0, 89.8], "hidden sizes": [32, 16], "total hs": 48, "epochs": 200},
    {"model": "model3", "R2": 85.75, "MSE": 1.00, "R2 std": 1.7, "R2 List": [85.0, 86.5], "hidden sizes": [32, 16], "total hs": 48, "epochs": 200}
]

results_table = pd.DataFrame(data=results)
latex_code = generate_latex_table(results_table, caption="Example Table", label="table:example")
print(latex_code)

