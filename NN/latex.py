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
                    latex_table_lines.append(['\\hline'] * len(row))
                row_span = 0
                multirow_model_col.append(model)
            previous_model = model
            latex_table_lines.append(row.tolist())

        if row_span > 0:
            latex_table_lines[-(row_span+1)][0] = f"\\multirow{{{row_span+1}}}{{*}}{{{previous_model}}}"
            latex_table_lines.append(['\\hline'] * len(row))

        escaped_table['model'] = multirow_model_col

    # Convert to LaTeX table using tabulate
    latex_table = tabulate(latex_table_lines, headers=escaped_table.columns, tablefmt='latex_raw', showindex=False)

    # Modify the column specifiers to use p{width} if needed
    column_format = "lrrrrrrr"  # Adjust columns types based on the number of columns

    # Replace the default column format with the desired one

    # Insert \hline at the beginning
    latex_table_lines = latex_table.splitlines()
    latex_table_lines.insert(1, '\\hline')
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

