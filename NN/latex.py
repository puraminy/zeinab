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

    # Apply the escaping function to all elements in the DataFrame
    escaped_table = table.applymap(lambda x: escape_latex_special_chars(str(x)) if isinstance(x, str) else x)

    # Process the first column for multirow
    if "model" in escaped_table:
        model_col = escaped_table['model'].tolist()
        multirow_model_col = []
        previous_model = None
        row_span = 0

        row_spans = []
        for i, model in enumerate(model_col):
            if model == previous_model:
                multirow_model_col.append('')
                row_span += 1
            else:
                if row_span > 0:
                    multirow_model_col[-(row_span+1)] = f"\\multirow{{{row_span+1}}}{{*}}{{{previous_model}}}"
                    row_spans.append(row_span)
                row_span = 0
                multirow_model_col.append(model)
            previous_model = model

        if row_span > 0:
            multirow_model_col[-(row_span+1)] = f"\\multirow{{{row_span+1}}}{{*}}{{{previous_model}}}"

        escaped_table['model'] = multirow_model_col

    # Convert to LaTeX table using tabulate
    latex_table = tabulate(escaped_table, headers='keys', tablefmt='latex_raw', showindex=False)

    # Insert \hline at the beginning and after each group of model names
    latex_table_lines = latex_table.splitlines()
    for i in range(len(latex_table_lines) - 2, 1, -1):
        if not latex_table_lines[i].startswith('\\multirow') and latex_table_lines[i+1].startswith('\\multirow'):
            latex_table_lines.insert(i+1, '\\hline')

    latex_table = '\n'.join(latex_table_lines)


    table_env = f"""
    \\begin{{table*}}
        \\centering
        {latex_table}
        \\caption{{{caption}}}
        \\label{{{label}}}
    \\end{{table*}}
    """

    return table_env
