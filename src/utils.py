import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
from typing import Any, Dict, List
def eyecolor_mapping(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    """
    Maps eye color values in a DataFrame column to predefined categories.

    Args:
        df (pd.DataFrame): The DataFrame containing the eye color column.
        column_name (str): The name of the column containing the eye color values.

    Returns:
        pd.DataFrame: The DataFrame with the eye color values mapped to categories.
    """
    df = df.copy()
    # Hardcoded list of strings to remove
    strings_to_remove = [
        'UNKNOWN', 'rs12913832 AA', 'GG', 'Rs12913832 aa', 'Gg', 
        'rs12913832 AG (they', 'Rs12913832 ag (they', 'rs12913832 GG', 
        'Rs12913832 gg', 'Brown center starburst, amber and olive green, with dark gray outer ring', 
        'Blue with yellow parts', 'blue-brown heterochromia', 'Red/Blood', 'Red/blood',
        'Blue with a yellow ring of flecks that make my eyes look green depending on the light or my  mood     '
    ] 

    colors_mapping = {
        "Brown": [
            'brown', 'Brown', 'Dark brown', 'Brown/black', 'braun', 
            'Brown-(green when external temperature rises)', 'Black',
            'Brown-Amber', 'Brown-amber', 'Amber - (yellow/ocre  brown)'
        ],
        "Green": ['green', 'Green', 'Green '],
        "Blue": ['blue', 'Blue', 'Dark blue'],
        "Hazel": ['Hazel', 'hazel', 'Hazel (brown/green)', 'Hazel/Light Brown', 
                    'Hazel/light brown', 'Hazel/Yellow', 'Hazel/yellow', 
                    'Hazel (light brown, dark green, dark blue)'],
        "Grey": ['Grey Brown', 'Grey brown'],
        "Brown-Green": ['Brown-green', 'brown-green', 'Green-brown', 'green-brown', 
                        'Brown - Brown and green in bright sunlight', 
                        'Brown - brown and green in bright sunlight', 
                        'Indeterminate brown-green with a subtle grey caste', 
                        'indeterminate brown-green with a subtle grey caste', 
                        'Olive-Brown ringing Burnt Umber-Brown', 
                        'Olive-brown ringing burnt umber-brown',
                        'Green with brown freckles'],
        "Blue-Green": ['blue-green', 'blue-green ', 'Blue-green', 'Blue-green ', 'Blue-green heterochromia',
                        'Light blue-green', 'Green-Hazel', 'Green-hazel', 
                        'Ambar-Green', 'Ambar-green', 'Green-gray', 'Green-blue outer ring and brown flecks around iris'],
        "Blue-Grey": ['blue-grey', 'Blue-grey', 'gray-blue', 'Gray-blue', 'Blue-grey with central heterochromia',
                        'blue grey', 'Blue grey', 'Blue-grey; broken amber collarette', 'Dark Grayish-Blue Eyes (like a stone)', 
                        'Blue-green; amber collarette, and gray-blue ringing'],
        "Mixed": [
            'mixed', 'Mixed', 'Light-mixed green', 'Light-mixed Green',
            'Blue, grey, green, changing', 'blue, grey, green, changing', 
            'Split - One side Dark Blue / Other side Light Blue and Green', 
            'Split - one side dark blue / other side light blue and green', 
            'Blue-green-grey', 'green-blue outer ring and brown flecks around iris'
        ]
    }   

    # Check if column exists in DataFrame
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' does not exist in the DataFrame")

    # Filter out rows with values in the specified column exactly matching any string in strings_to_remove
    df = df[~df[column_name].isin(strings_to_remove)]

    # Function to categorize a single eye color
    def categorize_color(color: str) -> str:
        for key, values in colors_mapping.items():
            if color in values:
                return key 

        raise ValueError(f"Could not categorize color '{color}'")

    # Apply the categorization
    df[column_name] = df[column_name].apply(categorize_color)

    return df

def plot_phenotype_distribution(df: pd.DataFrame, column_name: str) -> None:
    """
    Plots the distribution of a phenotype in a DataFrame column.

    Args:
        df (pd.DataFrame): The DataFrame containing the phenotype column.
        column_name (str): The name of the column containing the phenotype values.

    Returns:
        None
    """
    # Check if 'variation' column exists in the DataFrame
    if column_name not in df.columns:
        raise ValueError(f"{column_name} column not found in the DataFrame")

    # Plotting the histogram
    plt.figure(figsize=(10, 6))
    sns.histplot(df[column_name], kde=False)
    plt.title('Balance of Label Eyecolor')
    plt.xlabel('Variation')
    plt.ylabel('Frequency')
    plt.show()

def plot_rsin(df: pd.DataFrame) -> None:
    """
    Plots histograms for columns starting with 'rs' in a DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame containing the 'rs' columns.

    Returns:
        None
    """
    # Identify all 'rs' columns
    rs_columns = [col for col in df.columns if col.startswith('rs')]
    
    # Calculate the grid size needed for subplots
    n = len(rs_columns)
    ncols = 3  # Number of columns for the grid
    nrows = n // ncols + (n % ncols > 0)
    
    # Create a large figure to accommodate all the subplots
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, nrows * 5))
    axes = axes.flatten()  # Flatten the 2D array of axes

    for idx, rs_col in enumerate(rs_columns):
        # Filter out NaN values from the rs_column
        non_nan_values = df[rs_col].dropna()

        # Create the histogram on the appropriate subplot
        axes[idx].hist(non_nan_values, bins=10)  
        axes[idx].set_title(f'Histogram for {rs_col}')
        axes[idx].set_xlabel(rs_col)
        axes[idx].set_ylabel('Frequency')
        axes[idx].tick_params(labelrotation=90)

    # Hide any unused subplots
    for idx in range(n, len(axes)):
        fig.delaxes(axes[idx])

    plt.tight_layout()  # Adjust the layout
    plt.show()

def plot_rsin_eyecolor(df: pd.DataFrame) -> None:
    """
    Plots count plots for columns starting with 'rs' in a DataFrame, grouped by eye color.

    Args:
        df (pd.DataFrame): The DataFrame containing the 'rs' columns and 'eye_color' column.

    Returns:
        None
    """
    # Identify all 'rs' columns
    rs_columns = [col for col in df.columns if col.startswith('rs')]
    
    # Calculate the grid size needed for subplots
    n = len(rs_columns)
    ncols = 3  # You can choose a different number of columns for the grid
    nrows = n // ncols + (n % ncols > 0)
    
    # Create a large figure to accommodate all the subplots
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, nrows * 5))
    axes = axes.flatten()  # Flatten the 2D array of axes to simplify the loop

    for idx, rs_col in enumerate(rs_columns):
        # Filter out NaN values from the rs_column
        filtered_df = df.dropna(subset=[rs_col])
        
        # Create the count plot on the appropriate subplot
        sns.countplot(x=rs_col, hue='eye_color', data=filtered_df, ax=axes[idx])
        axes[idx].set_title(f'Count Plot for {rs_col}')
        axes[idx].set_xlabel(rs_col)
        axes[idx].set_ylabel('Count')
        axes[idx].tick_params(labelrotation=90)
        axes[idx].legend(title='Eye Color')

    # Hide any unused subplots
    for idx in range(n, len(axes)):
        fig.delaxes(axes[idx])

    plt.tight_layout()  # Adjust the layout
    plt.show()

def get_unique_rs_values(df: pd.DataFrame) -> None:
    """
    Prints the unique values across all 'rs' columns in a DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame containing the 'rs' columns.

    Returns:
        None
    """
    # Select columns that start with 'rs'
    rs_columns = df.columns[df.columns.str.startswith('rs')]

    # Get unique values across all 'rs' columns
    unique_values = pd.unique(df[rs_columns].values.ravel('K'))

    # Print the unique values
    print(unique_values)

def model_comparison(models_dict: Dict[str, Dict[str, Any]], criteria: str) -> pd.DataFrame:
    """
    Compares the performance of models based on a given criteria.

    Args:
        models_dict (Dict[str, Dict[str, Any]]): A dictionary containing the model names as keys and their details as values.
        criteria (str): The criteria used for comparison.

    Returns:
        pd.DataFrame: A DataFrame containing the comparison results.
    """
    # Prepare lists to store results
    models_list = []
    accuracy_normal_list = []
    accuracy_criteria_list = []
    dominant_model_list = []

    # Iterate through the dictionary and find models matching the criteria
    for model_name in models_dict.keys():
        if criteria in model_name:
            # Extract the base model name by removing the criteria
            base_model_name = model_name.replace(criteria, '')
            model_with_criteria = model_name
            normal_model = base_model_name

            # Check if both normal and criteria models exist
            if model_with_criteria in models_dict and normal_model in models_dict:
                acc_normal = models_dict[normal_model]['test_acc']
                acc_criteria = models_dict[model_with_criteria]['test_acc']
                models_list.append(base_model_name)
                accuracy_normal_list.append(acc_normal)
                accuracy_criteria_list.append(acc_criteria)

                # Determine which model is dominant based on accuracy
                dominant_model = 'Normal' if acc_normal >= acc_criteria else f'{criteria}'
                dominant_model_list.append(dominant_model)

    # Create DataFrame
    comparison_df = pd.DataFrame({
        'Model': models_list,
        'Accuracy_Normal': accuracy_normal_list,
        f'Accuracy_{criteria}': accuracy_criteria_list,
        'Dominant_Model': dominant_model_list
    })

    return comparison_df

def compare_accuracy(models: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """
    Compares the accuracy of models and returns a DataFrame sorted by accuracy.

    Args:
        models (Dict[str, Dict[str, Any]]): A dictionary containing the model names as keys and their details as values.

    Returns:
        pd.DataFrame: A DataFrame containing the model names and their accuracies, sorted by accuracy in descending order.
    """
    # Filter models with '_NO_NAN' and without '_WEIGHTED' in their names
    filtered_models = {k: v for k, v in models.items() if "_NO_NAN" in k and "_WEIGHTED" not in k}

    # Extract and compare accuracies
    accuracies = {model_name: details["test_acc"] for model_name, details in filtered_models.items()}
    
    # Create a DataFrame
    accuracies_df = pd.DataFrame(list(accuracies.items()), columns=['Model', 'Accuracy'])

    # Sort the DataFrame based on accuracy in descending order
    accuracies_df = accuracies_df.sort_values(by='Accuracy', ascending=False).reset_index(drop=True)

    return accuracies_df
