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
    escaped_table = table.map(lambda x: escape_latex_special_chars(str(x)) if isinstance(x, str) else x)

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
                    multirow_model_col[-(row_span+1)] = f"\\hline\n\\multirow{{{row_span+1}}}{{*}}{{{previous_model}}}"
                    row_spans.append(i)
                row_span = 0
                multirow_model_col.append("\\hline\n" + model)
            previous_model = model

        if row_span > 0:
            multirow_model_col[-(row_span+1)] = f"\\hline\n\\multirow{{{row_span+1}}}{{*}}{{{previous_model}}}"

        escaped_table['model'] = multirow_model_col

    # Convert to LaTeX table using tabulate
    latex_table = tabulate(escaped_table, headers='keys', tablefmt='latex_raw', showindex=False)

    table_env = f"""
    \\begin{{table*}}[h]
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
    {"model": "model2", "R2": 88.38, "MSE": 0.98, "R2 std": 1.8, "R2 List": [87.0, 89.8], "hidden sizes": [32, 16], "total hs": 48, "epochs": 200},
    {"model": "model3", "R2": 85.75, "MSE": 1.00, "R2 std": 1.7, "R2 List": [85.0, 86.5], "hidden sizes": [32, 16], "total hs": 48, "epochs": 200},
    {"model": "model3", "R2": 85.75, "MSE": 1.00, "R2 std": 1.7, "R2 List": [85.0, 86.5], "hidden sizes": [32, 16], "total hs": 48, "epochs": 200},
    {"model": "model4", "R2": 85.75, "MSE": 1.00, "R2 std": 1.7, "R2 List": [85.0, 86.5], "hidden sizes": [32, 16], "total hs": 48, "epochs": 200},
    {"model": "model4", "R2": 85.75, "MSE": 1.00, "R2 std": 1.7, "R2 List": [85.0, 86.5], "hidden sizes": [32, 16], "total hs": 48, "epochs": 200}
]

results_table = pd.DataFrame(data=results)
latex_code = generate_latex_table(results_table, caption="Example Table", label="table:example")
with open("test.tex", "w") as f:
    print(latex_code, file=f)

