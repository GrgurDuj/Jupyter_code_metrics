import os
import pdb
import nbformat
import ast
import pprint
import seaborn as sns
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def process_notebook(file_path):
    code_cells = get_codecells_from_notebook(file_path)
    markdown_cells = get_markdowncells_from_notebook(file_path)

    lines = get_lines(code_cells)
    func_num = get_function_number(code_cells)
    attr_num = get_attributes(code_cells)
    complexity = get_complexity(code_cells)
    com_perc = get_comment(code_cells)
    num_mkdn_words = get_number_mkdn_words(markdown_cells)
    func_calls = get_num_func_calls(code_cells)
    
    return {
        "file_path": file_path,
        "num_code_cells": len(code_cells),
        "num_markdown_cells": len(markdown_cells),
        "lines": lines,
        "func_num": func_num,
        "attr_num": attr_num,
        "complexity": complexity,
        "com_perc": com_perc,
        "mean_lines_codecells": lines / len(code_cells),
        "num_mkdn_words": num_mkdn_words,
        "mean_mkdn_words": num_mkdn_words / len(markdown_cells),
        "func_calls": func_calls,
    }

def get_lines(code_cells):
    lines = 0
    for cell in code_cells:
        source_lines = cell.source.split('\n')
        lines += len([line for line in source_lines if line.strip() != ''])
    return lines

def get_function_number(code_cells):
    func_num = 0
    for cell in code_cells:
        tree = ast.parse(cell.source)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                func_num += 1
    return func_num

def get_codecells_from_notebook(file_path):
    with open(file_path, 'r') as file:
        notebook = nbformat.read(file, as_version=4)

    code_cells = [cell for cell in notebook.cells if cell.cell_type == 'code']

    return code_cells

def get_markdowncells_from_notebook(file_path):
    with open(file_path, 'r') as file:
        notebook = nbformat.read(file, as_version=4)

    markdowncells = [cell for cell in notebook.cells if cell.cell_type == 'markdown']

    return markdowncells

def get_complexity(code_cells):
    complexity = 0
    for cell in code_cells:
        tree = ast.parse(cell.source)
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.BoolOp)):
                complexity += 1
    return complexity + 1
    
def get_attributes(code_cells):
    attr_num = 0
    for cell in code_cells:
        tree = ast.parse(cell.source)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                attr_num += len(node.args.args)
    return attr_num

def get_comment(code_cells):
    total_lines = 0
    comment_lines = 0
    for cell in code_cells:
        source_lines = [line for line in cell.source.split('\n') if line.strip() != '']
        total_lines += len(source_lines)
        comment_lines += len([line for line in source_lines if line.strip().startswith('#')])
    
    if total_lines == 0:
        return 0
    else:
        return int((comment_lines / total_lines) * 100)

def get_mean_lines(code_cells):
    lines = 0
    for cell in code_cells:
        source_lines = cell.source.split('\n')
        lines += len([line for line in source_lines if line.strip() != ''])# and not line.strip().startswith('#')])
    return lines / len(code_cells)

def get_number_mkdn_words(markdown_cells):
    words = 0
    for cell in markdown_cells:
        words += len(cell.source.split())
    return words

def get_num_func_calls(code_cells):
    func_calls = 0
    for cell in code_cells:
        tree = ast.parse(cell.source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func_calls += 1
    return func_calls

def pretty_print(data):
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(data)

def process_examples(group):
    results = []
    raw_results = []
    path = os.path.join(os.getcwd(), "assignments", group)

    for year in os.listdir(path):
        print(year, "year")
        for instance in os.listdir(os.path.join(path, year)):
            field_values = {}
            field_sums = {}
            field_counts = {}
            if instance.endswith(".pdf") or instance.endswith(".tsv"):
                continue
            for sub in os.listdir(os.path.join(path, year, instance)):
                print(sub, "sub")
                if sub.endswith(".ipynb"):
                    try:
                        notebook_results = process_notebook(os.path.join(path, year, instance, sub))
                        for field, value in notebook_results.items():
                            if field == "file_path":
                                continue
                            if field not in field_values:
                                field_values[field] = []
                            field_values[field].append(value)
                            field_sums[field] = int(field_sums.get(field, 0)) + int(value)
                            field_counts[field] = int(field_counts.get(field, 0)) + 1
                    except Exception as e:
                        print(e)
                        print(f"Notebook {path} failed to process")
                        continue
            averages = {field: total / field_counts[field] for field, total in field_sums.items()}
            results.append((year, averages))
            raw_results.append((year, field_values))

    # Convert your data into a DataFrame
    df = pd.DataFrame([item[1] for item in results])
    df['year'] = [item[0] for item in results]
    
    # Group by year and calculate the mean
    values_grouped = df.groupby('year').mean()

    df = pd.DataFrame([item[1] for item in raw_results])
    df['year'] = [item[0] for item in results]

    raw_values_grouped = df.groupby('year').sum()

    return values_grouped, raw_values_grouped
    
def generate_correlation_heatmap(cse_results, non_results):
    # Concatenate the two dataframes along the columns axis
    combined_df = pd.concat([cse_results.add_suffix('_cse'), non_results.add_suffix('_non')], axis=1)

    # Compute correlations between pairs of fields
    correlations = combined_df.corr()

    # Compute absolute values of correlations between pairs of fields
    abs_correlations = correlations.abs()

    # Create a heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(abs_correlations, annot=True, cmap='coolwarm')

    plt.savefig("heatmap.png")  # Save the figure as a PNG file

    print(correlations)

def compute_confidence_intervals(df, confidence=0.95):
    confidence_intervals = {}

    for year in df.index:
        for metric in df.columns:
            data = df.loc[year, metric]
            mean = np.mean(data)
            se = stats.sem(data)
            interval = stats.t.interval(confidence, len(data) - 1, loc=mean, scale=se)
            confidence_intervals[(year, metric)] = interval

    return confidence_intervals

def compute_t_tests(raw_cse_results, raw_non_results):
    t_tests = {}

    for year in raw_cse_results.index:
        for metric in raw_cse_results.columns:
            cse_data = raw_cse_results.loc[year, metric]
            non_data = raw_non_results.loc[year, metric]
            t_tests[(year, metric)] = stats.ttest_ind(cse_data, non_data)

    return t_tests

def compute_cohens_d(raw_cse_results, raw_non_results):
    cohens_d = {}

    for year in raw_cse_results.index:
        for metric in raw_cse_results.columns:
            cse_data = raw_cse_results.loc[year, metric]
            non_data = raw_non_results.loc[year, metric]
            pooled_var = (len(cse_data) - 1 * np.var(cse_data) + (len(non_data) -1) * np.var(non_data)) / (len(cse_data) + len(non_data) - 2)
            cohens_d[(year, metric)] = (np.mean(cse_data) - np.mean(non_data)) / (np.sqrt(pooled_var))

    return cohens_d

def generate_bar_plots(aggregated_results, cse_results, non_results, cse_confidence_intervals, non_confidence_intervals):
    # Get all field names
    fields = set()
    for group_results in aggregated_results.values():
        df_columns = group_results.select_dtypes(include=[np.number]).columns
        fields.update(df_columns)

    # Get all years
    years = sorted(set(cse_results.index.tolist() + non_results.index.tolist()))
    # Create a histogram for each field
    plt.figure(figsize=(10, 12))  # Set the figure size

    for i, field in enumerate(fields):
        plt.subplot(6, 2, i+1)  # Create a subplot for each field
        bar_width = 0.35
        index = np.arange(len(years))

        for j, (group, group_results) in enumerate(aggregated_results.items()):
            values = [next((year_results.get(field, 0) for year, year_results in group_results.iterrows() if year == year_), 0) for year_ in years]
            plt.bar(index + j * bar_width, values, bar_width, label=group)
            #pdb.set_trace()
            # Add confidence intervals
            for year_index, year in enumerate(years):
                if group == "cse":
                    confidence_interval = cse_confidence_intervals.get((str(year), field))
                else:
                    confidence_interval = non_confidence_intervals.get((str(year), field))
        
                if confidence_interval:
                    plt.errorbar(year_index + j * bar_width, values[year_index], yerr=np.abs(np.subtract(confidence_interval[0], values[year_index])), fmt='none', color='black', capsize=3)

        plt.xlabel('Year')
        plt.ylabel(field)
        plt.title('Comparison of ' + field)
        plt.xticks(index + bar_width / 2, years)
        plt.legend()

    plt.tight_layout()
    plt.savefig("bar_plots.png")  # Save the figure as a PNG file
    #plt.show()

def plot_p_d(significant_metrics, significant_pvalues, cohens_d_values):
    cohens_d_values = [abs(d) for d in cohens_d_values]
    # Normalize Cohen's d values to fit within the range of p-values
    max_dvalue = max(cohens_d_values)
    max_pvalue = max(significant_pvalues)
    normalized_cohens_d = np.interp(cohens_d_values, (0, max_dvalue), (0, max_pvalue))

    # Plot the significant p-values and normalized Cohen's d values on the same axis
    fig, ax = plt.subplots(figsize=(10, 6))

    # Bar height for grouped bars
    bar_height = 0.35
    index = np.arange(len(significant_metrics))

    # Plot p-values
    ax.set_xlabel('p-values')
    ax.barh(index, significant_pvalues, color='blue', height=bar_height, label='p-values')

    # Create a secondary x-axis for Cohen's d
    ax2 = plt.gca().twiny()
    ax2.set_xlabel("Cohen's d")
    ax2.barh(index + bar_height, normalized_cohens_d, color='skyblue', height=bar_height, label="Cohen's d")

    # Set labels for axes
    plt.ylabel('Metrics')

    # Set x-axis ticks and labels for Cohen's d
    ax2.set_xlim(0, max_pvalue)
    ax2.set_xticks(np.round(np.linspace(0, max_pvalue, 5), 3))
    ax2.set_xticklabels(np.round(np.linspace(0, max_dvalue, 5), 3))

    # Set y-axis ticks and labels
    plt.yticks(index + bar_height / 2, significant_metrics)

    # Create separate legend handles and labels
    legend_handles = [Line2D([0], [0], color='blue', lw=6), Line2D([0], [0], color='skyblue', lw=6)]
    legend_labels = ['p-values', "Cohen's d"]

    # Set legend
    plt.legend(legend_handles, legend_labels, loc='lower right')

    plt.tight_layout()
    plt.savefig("p_d.png")  # Save the figure as a PNG file
    plt.show()


def main():
    cse_results, raw_cse_results = process_examples("cse")
    non_results, raw_non_results = process_examples("non")

    aggregated_results = {
        "cse": cse_results,
        "non": non_results
    }
    pretty_print(aggregated_results)

    generate_correlation_heatmap(cse_results, non_results)

    cse_confidence_intervals = compute_confidence_intervals(raw_cse_results)
    non_confidence_intervals = compute_confidence_intervals(raw_non_results)

    print("CSE confidence intervals:")
    pretty_print(cse_confidence_intervals)
    print("Non-CSE confidence intervals:")
    pretty_print(non_confidence_intervals)

    generate_bar_plots(aggregated_results, raw_cse_results, raw_non_results, cse_confidence_intervals, non_confidence_intervals)

    cohens_ds = compute_cohens_d(raw_cse_results, raw_non_results)
    print("Cohen's d:")
    pretty_print(cohens_ds)

    t_tests = compute_t_tests(raw_cse_results, raw_non_results)
    print("T-tests:")
    pretty_print(t_tests)

    significant_pvalues = []
    significant_metrics = []
    significant_d = []

    print("T-tests with p-value less than 0.05:")
    for metric, (statistic, pvalue) in t_tests.items():
        if pvalue < 0.05:
            print(f"{metric}: t = {statistic}, p = {pvalue}")
            significant_pvalues.append(pvalue)
            significant_metrics.append(str(metric))  # Convert to a string
            significant_d.append(cohens_ds[metric])
    
    significant_pvalues = significant_pvalues[::-1]
    significant_metrics = significant_metrics[::-1]
    significant_d = significant_d[::-1]

    plot_p_d(significant_metrics, significant_pvalues, significant_d)
    
if __name__ == "__main__":
    main()
