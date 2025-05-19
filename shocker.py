import numpy as np
import pandas as pd
import os

# Typhoon Damage Data from ADB Paper
typhoon_damage = {
    "Region":   [5,     10,     20,     50],
    "NCR":      [0.5,   0.67,   0.76,   0.81],
    "CAR":      [3.17,	8.45,	10.8,	12.03],
    "I":        [3.67,  5.77,   6.81,   7.43],
    "II":       [1.6,	13.13,	17.85,	20.12],
    "III":      [2.19,	3.07,	3.73,	4.35],
    "IVA":      [1.46,	2.07,	2.69,	3.54],
    "IVB":      [3.91,	5.49,	6.25,	6.68],
    "V":        [9.37,	11.76,	14.74,	19.85],
    "VI":       [4.085,	5.165,	5.565,	5.77],
    "VII":      [1.14,	5.05,	6.48,	7.09],
    "VIII":     [7.58,	14.5,	17.96,	20.04],
    "IX":       [0.09,	0.09,	0.09,	0.09],
    "X":        [4.67,	8.94,	11.08,	12.36],
    "XI":       [1.21,	1.3,	1.34,	1.37],
    "XII":      [0.75,	0.81,	0.84,	0.85],
    "XIII":     [5.32,	8.34,	9.84,	10.75],
    "BARMM":    [1.63,	1.69,	1.72,	1.74]
}
reduction_df = pd.DataFrame(typhoon_damage).T
reduction_df.index.name = "Region"
reduction_df.columns = [5, 10, 20, 50] 

# Load the Excel file
file_path = "GRDP.xlsx"
df = pd.read_excel(file_path, sheet_name="Sheet1", index_col=0, header=0)
#print(df.head())

sectors = df.columns.drop("TID").tolist()
# print ("Sectors:", sectors)



def get_shocks(return_period: int, affected_regions: list):
    # Returns a dictionary of the shock values for each sector
    # for the given return period and affected regions

    if return_period not in reduction_df.columns:
        raise ValueError(f"Invalid return period. Choose from: {list(reduction_df.columns)}")

    # copy original dataframe to be written in a spreasheet
    shocked_df = df.copy()

    # get the damage multipliers for the given return period 
    multipliers = 1 - (reduction_df[return_period] / 100)
    
    # for each region, multiply the original values by the multipliers
    for region in affected_regions:
        if region in multipliers.index:
            # print(f"Applying multiplier for {region}: {multipliers[region]}")

            # multiply the original values in each sector by the multipliers
            shocked_df.loc[region] = df.loc[region] * multipliers[region]

            # print("Value Deducted:", df.loc[region] * (1 - multipliers[region]))
    
    # Add new column
    #shocked_df["New TID"] = shocked_df.drop("TID", axis=1).sum(axis=1)

    # Add new rows 
    # gets the new tpi by summing the shocked values
    shocked_df.loc["New TPI"] = shocked_df.drop("TPI").sum()

    # gets the total damage by subtracting the new tpi from the original tpi (same with adding all dedcuted values)
    shocked_df.loc["Damage"] = shocked_df.loc["TPI"] - shocked_df.loc["New TPI"]

    # Factor is the ratio of the new tpi to the original tpi
    shocked_df.loc["Factor"] = shocked_df.loc["New TPI"] / shocked_df.loc["TPI"]
    
    # To output:
    sector_shocks = {
        sector : float(shocked_df.loc["Factor", sector]) for sector in sectors
    }
    # for element in sector_shock:
    #     print(f"{element}: {sector_shock[element]}")

    # Write the shocked data to a new Excel file
    # output_file = f"shocked_values_return_period_{return_period}.xlsx"
    # shocked_df.to_excel(output_file)
    # print(f"Shocked data saved to: {output_file}")

    return sector_shocks

# aff = ["NCR", 
#        "CAR", 
#        "I", 
#        "II", 
#        "III", 
#        "IVA", 
#        "IVB", 
#        "V", 
#        "VI", 
#        "VII", 
#        "VIII", 
#        "IX", 
#        "X", 
#        "XI", 
#        "XII", 
#        "XIII", 
#        "BARMM"] 

# get_shocks(50, aff)