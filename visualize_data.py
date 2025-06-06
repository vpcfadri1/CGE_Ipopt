import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

def export_pyomo_variables_to_excel(model, variables_names, filename):
    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
        for var_name in variables_names:
            if not hasattr(model, var_name):
                print(f"No variable named {var_name} in the model")
                return
            var = getattr(model, var_name)
            data = {k: v.value for k, v in var.items()}

            # check if scalar
            if len(data) == 1:
                value = data[None]
                df = pd.DataFrame([[value]])
            else:
                df = pd.DataFrame([(k, v) for k, v in data.items()])
            
            df.to_excel(writer, sheet_name=var_name[:31], index=False)
    return


def shocked_variables_to_excel(model, variable_names, old_filename, new_filename):
    with pd.ExcelWriter(new_filename, engine='openpyxl') as writer:
        for var_name in variable_names:
            var = getattr(model, var_name)
            data = {k: v.value for k, v in var.items()}
            data_str_keys = {str(k): v for k, v in data.items()}

            old_df = pd.read_excel(old_filename, sheet_name=var_name)
            # check if scalar
            if len(data) == 1:
                updated_value = data[None]
                original_value = old_df.iloc[0, 0]
                old_df["Updated"] = [updated_value]
                percent_change = (updated_value - original_value) / original_value * 100
                old_df["%Change"] = [percent_change]
            else:
                index_col = old_df.iloc[:, 0]
                original_col = old_df.iloc[:, 1]
                old_df["key_tuple"] = list(zip(old_df.iloc[:, 0], old_df.iloc[:, 1]))
                old_df["Updated"] = index_col.map(data_str_keys)
                old_df["%Change"] = 100 * (old_df["Updated"] - original_col) / original_col
                old_df = old_df.drop(columns=["key_tuple"])
                old_df = old_df.rename(columns={0: "Sector", 1: "Initial Equilibrium", "Updated": "Baseline Model Run", "%Change": "Change (%)"},)
            
            old_df.to_excel(writer, sheet_name=var_name[:31], index=False)

    return



def create_charts(file_name, var_name, name, two_vars=False):
    # Create a bar chart comparing two variables from an Excel file

    df = pd.read_excel(file_name, sheet_name=var_name, header=0)

    x = df['Sector']
    y1 = df['Initial Equilibrium']
    y2 = df['Baseline Model Run']
    y3 = df['Change (%)']
    width = 0.35  
    x_pos = range(len(x))  

    fig, ax = plt.subplots(figsize=(12, 6))

    if two_vars: # For compare two variables like for Q
        ax.bar([p - width/2 for p in x_pos], y1, width, label='Initial Equilibrium', )
        ax.bar([p + width/2 for p in x_pos], y2, width, label='Shocked Equilibrium', )
        ax.set_ylabel("Gross Domestic Output in Millions of Pesos")
        ax.legend()

    else:       # To see the change in price
        ax.bar(x_pos, y3, width, label='Change in Price (%)')
        ax.axhline(0, color='black', linewidth=1)
        ax.set_ylabel("Change in Price (%)")
        

    ax.set_title(name)
    ax.set_xlabel("Sectors")
    ax.set_xticks(list(x_pos))
    ax.set_xticklabels(x, rotation=45)

    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=False))
    ax.ticklabel_format(style='plain', axis='y')  
    ax.get_yaxis().get_offset_text().set_visible(False) 
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()