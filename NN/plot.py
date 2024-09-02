import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score

def escape_latex_special_chars(text):
    return (text.replace('_', '-')
                .replace(' ', '-')
                .replace('%', '\\%')
                .replace('$', '\\$')
                .replace('&', '\\&')
                .replace('#', '\\#')
                .replace('{', '\\{')
                .replace('}', '\\}'))

def plot_model_performance(results_table, latex_filename='plots_latex.tex'):
    # Ensure the "plots" directory exists
    os.makedirs('plots', exist_ok=True)

    # LaTeX code storage
    latex_code = ''

    # Plot performance for each model and hidden size
    for model_name in results_table['model'].unique():
        # Filter the results for the current model
        model_results = results_table[results_table['model'] == model_name]

        # Get unique hidden sizes
        unique_hidden_sizes = model_results['total hs'].unique()
        
        for hidden_size in unique_hidden_sizes:
            # Filter results for the current hidden size
            hs_results = model_results[model_results['total hs'] == hidden_size]

            # Sort results by epochs
            sorted_results = hs_results.sort_values(by='epochs')

            # Extract epochs and R2 values
            epochs = sorted_results['epochs'].to_numpy()
            r2_values = sorted_results['R2'].to_numpy()

            # Plot R2 vs. Number of Epochs for the current model and hidden size
            plt.figure(figsize=(10, 6))
            plt.plot(epochs, r2_values, 'o-', label=f'R2 (Hidden Size: {hidden_size})')
            plt.title(f'{model_name} - R2 vs. Number of Epochs (Hidden Size: {hidden_size})')
            plt.xlabel('Number of Epochs')
            plt.ylabel('R2')
            plt.grid(True)
            plt.legend()
            file_name = f'{model_name}_R2_vs_Num_of_Epochs_HiddenSize_{hidden_size}.png'
            file_name = escape_latex_special_chars(file_name)
            plot_filename = "plots/" + file_name # os.path.join('plots', file_name)
                    
            plt.savefig(plot_filename)
            plt.close()

            # Add LaTeX code for this plot
            plot_caption = f'{model_name} - R2 vs. Number of Epochs (Hidden Size: {hidden_size})'
            latex_code += f"""
\\begin{{figure}}[H]
    \\centering
    \\includegraphics[width=0.8\\textwidth]{{{plot_filename}}}
    \\caption{{{plot_caption}}}
\\end{{figure}}
"""
    # Save LaTeX code to a file
    with open(latex_filename, 'w', encoding='utf-8') as f:
        f.write(latex_code)



def plot_results(predictions_np, y_test, title, file_name, latex_filename='plots_preds_latex.tex', show_plot=False):
    # Ensure the "plots" directory exists
    os.makedirs('plots_preds', exist_ok=True)
    file_name = escape_latex_special_chars(file_name)
    
    # Convert predictions and y_test to NumPy arrays
    y_test_np = y_test.numpy()
    # Calculate R-squared
    r2 = r2_score(y_test_np, predictions_np)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(y_test_np, y_test_np, color='green', label='Actual', alpha=0.5)
    ax.scatter(y_test_np, predictions_np, color='blue', label='Predicted', alpha=0.5)

    # Plot the 45-degree line where x = y
    min_val = min(np.min(y_test_np), np.min(predictions_np))
    max_val = max(np.max(y_test_np), np.max(predictions_np))
    ax.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label='45-Degree Line')


    ax.set_title(title + ' ' +  f' R-Squared:{r2:.2f}')
    ax.set_xlabel('Actual Values')
    ax.set_ylabel('Predicted Values')
    ax.legend()
    ax.grid(True)
    if show_plot is True:
        plt.show()
    # Save the image of plot in plots folder
    image_path = "plots_preds/" + file_name # os.path.join("plots_preds", file_name)
    fig.savefig(image_path, format="png")
    plt.close()

    # Generate LaTeX code for this plot
    plot_caption = f'{title} - R-Squared: {r2:.2f}'
    latex_code = f"""
\\begin{{figure}}[H]
    \\centering
    \\includegraphics[width=0.8\\textwidth]{{{image_path}}}
    \\caption{{{plot_caption}}}
\\end{{figure}}
"""

    # Append LaTeX code to the existing file
    with open(latex_filename, 'w') as f:
        f.write(latex_code)



