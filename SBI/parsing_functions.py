import pickle
import os
import re

def load_input_data(input_file):
    with open(input_file, 'rb') as file:
        dataset = pickle.load(file)
    
    #print(dataset)  # Print the dataset to understand its structure
    
    # Initialize empty lists to store theta and x values
    theta_tensor = dataset['theta']
    x_tensor = dataset['x']
    return x_tensor, theta_tensor

def extract_specs(dataset_name):
    """
    Extract specifications from a dataset filename.
    
    Parameters:
    dataset_name (str): The dataset filename.
    
    Returns:
    dict: A dictionary containing the extracted specifications and type.
    """
    # Check for 'merged_corr_' or 'merged_svr_' prefix
    if 'merged_corr_' in dataset_name:
        prefix = 'corr'
        specs_str = dataset_name.replace('merged_corr_', '').split('dataset_')[-1].split('.pickle')[0]
    elif 'merged_svr_' in dataset_name:
        prefix = 'svr'
        specs_str = dataset_name.replace('merged_svr_', '').split('dataset_')[-1].split('.pickle')[0]
    else:
        prefix = None
        specs_str = dataset_name.replace('merged_', '').split('dataset_')[-1].split('.pickle')[0]
    
    # Regular expression to match each specification
    pattern = r'(\d+\.?\d*[eE]?[+-]?\d*)'
    
    # Extract parameters from the specifications
    specs_list = re.findall(pattern, specs_str)
    
    # Convert the extracted strings to appropriate types
    num_simulations = float(specs_list[0])
    Npts = float(specs_list[1])
    dt = float(specs_list[2])
    oversampling = int(specs_list[3])
    prerun = float(specs_list[4])
    
    return {
        'num_simulations': num_simulations,
        'Npts': Npts,
        'dt': dt,
        'oversampling': oversampling,
        'prerun': prerun,
        'prefix': prefix
    }

def save_pickle_data(data, filename, folder_path):
    """
    Save a dataset to a specified folder with a unique filename.
    
    Parameters:
    data (dict): The data to be saved, typically containing 'theta' and 'x'.
    filename (str): The base filename for the dataset.
    folder_path (str): The path to the folder where the dataset will be saved.
    
    Returns:
    str: The full path of the saved dataset.
    """
    # Create directory to save dataset if it doesn't exist
    if not os.path.isdir(folder_path):
        os.makedirs(folder_path)
    
    # Extract the base name and extension from the filename
    base_name, ext = os.path.splitext(filename)
    if not ext:
        ext = '.pickle'  # default to .pickle if no extension is provided
    
    # Check if file already exists and create a unique filename if it does
    counter = 1
    dataset_name = base_name + ext
    dataset_path = os.path.join(folder_path, dataset_name)
    while os.path.exists(dataset_path):
        dataset_name = f'{base_name}_{counter}{ext}'
        dataset_path = os.path.join(folder_path, dataset_name)
        counter += 1
    
    # Save dataset
    with open(dataset_path, 'wb') as file:
        pickle.dump(data, file)
    
    print(f'Saved dataset at {dataset_path}\n')
    #return dataset_path



# Filter files that have the same specs except for num_simulations and are not already merged
def specs_match(filename, target_Npts, target_dt, target_oversampling, target_prerun):
    if filename.startswith("merged_"):
        return False
    try:
        specs = extract_specs(filename)
        return (specs['Npts'] == target_Npts and
                specs['dt'] == target_dt and
                specs['oversampling'] == target_oversampling and
                specs['prerun'] == target_prerun)
    except (IndexError, ValueError):
        return False