import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

def eyecolor_mapping(df, column_name):
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
    def categorize_color(color):
        for key, values in colors_mapping.items():
            if color in values:
                return key 

        raise ValueError(f"Could not categorize color '{color}'")

    # Apply the categorization
    df[column_name] = df[column_name].apply(categorize_color)

    return df

def plot_phenotype_distribution(df, column_name):
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

def plot_rsin(df):
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

def plot_rsin_eyecolor(df):
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

def get_unique_rs_values(df):
    # Select columns that start with 'rs'
    rs_columns = df.columns[df.columns.str.startswith('rs')]

    # Get unique values across all 'rs' columns
    unique_values = pd.unique(df[rs_columns].values.ravel('K'))

    # Print the unique values
    print(unique_values)

def plot_images(folder_path):
    # List all files in the folder
    files = os.listdir(folder_path)

    # Filter out files that are images (assuming common image file extensions)
    image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]

    # Determine the number of rows needed in the subplot
    num_rows = len(image_files) // 2 + (len(image_files) % 2 > 0)

    # Adjust the size of the figure (width, height)
    fig_width = 15 
    fig_height = num_rows * 5 

    # Create subplots
    fig, axes = plt.subplots(num_rows, 2, figsize=(fig_width, fig_height))
    axes = axes.flatten()

    # Plot each image
    for i, file in enumerate(image_files):
        img_path = os.path.join(folder_path, file)
        img = mpimg.imread(img_path)
        axes[i].imshow(img)
        axes[i].set_title(file.replace('_', ' ').replace('.png', ''))
        axes[i].axis('off')

    # Hide unused axes if any
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()

