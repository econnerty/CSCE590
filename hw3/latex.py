import pandas as pd

# Load the CSV files for each case into DataFrames
case1_df = pd.read_csv('problem2_case1.csv')
case2_df = pd.read_csv('problem2_case2.csv')

# Function to convert a DataFrame into a LaTeX table
def df_to_latex(df, label, caption):
    latex_table = df.to_latex(index=False, caption=caption, label=label)
    return latex_table

# Convert each DataFrame to a LaTeX table
latex_table_case1 = df_to_latex(case1_df, 'tab:case1', 'Case 1 results for the quadratic function.')
latex_table_case2 = df_to_latex(case2_df, 'tab:case2', 'Case 2 results for the quadratic function.')

# Return LaTeX table strings
latex_table_case1, latex_table_case2

print(latex_table_case2)
