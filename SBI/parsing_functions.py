import pickle
import os
import re
import torch
from typing import Optional
#import numpy as np

def load_inference_data(input_file):
    with open(input_file, 'rb') as file:
        dataset = pickle.load(file)
    
    # Initialize empty lists to store theta and x values
    theta_tensor = dataset['theta']
    x_tensor = dataset['x']
    return x_tensor, theta_tensor

def extract_specs(file_path):
    """
    Extract specifications from a dataset file.
    
    Parameters:
    file_path (str): The path to the dataset file.
    
    Returns:
    dict: A dictionary containing the extracted specifications and metadata.
    """
    filename = os.path.basename(file_path)
    
    # Determine prefix based on filename
    if filename.startswith('merged_corr_'):
        prefix = 'merged_corr'
    elif filename.startswith('merged_svr_'):
        prefix = 'merged_svr'
    elif filename.startswith('corr_'):
        prefix = 'corr'
    elif filename.startswith('svr_'):
        prefix = 'svr'
    else:
        prefix = None
    
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    
    specs = {
        'num_simulations': data.get('num_simulations', None),
        'Npts': data.get('Npts', None),
        'dt': data.get('dt', None),
        'oversampling': data.get('oversampling', None),
        'prerun': data.get('prerun', None),
        'low_tensor': data.get('low_tensor', None),
        'high_tensor': data.get('high_tensor', None),
        'data_type': data.get('data_type', None),
        'prefix': prefix,
        'features': data.get('features', None)
    }
    
    return specs


def save_pickle_data(data, folder_path, prefix=None, merge=False):
    """
    Save a dataset to a specified folder with a unique filename or overwrite if merging.
    
    Parameters:
    data (dict): The data to be saved, typically containing 'theta', 'x', or 'posterior', and metadata.
    folder_path (str): The path to the folder where the dataset will be saved.
    prefix (str or None): The prefix for the dataset filename. If None, no prefix is added.
    merge (bool): If True, overwrite the existing file with the same specs instead of creating a new one.
    
    Returns:
    str: The full path of the saved dataset.
    """
    # Create directory to save dataset if it doesn't exist
    if not os.path.isdir(folder_path):
        os.makedirs(folder_path)
    
    # Determine base name based on the data content
    if 'posterior' in data:
        dataset_type = 'posterior'
    else:
        dataset_type = 'dataset'
    
    # Construct the base filename from metadata
    base_name = (f"{dataset_type}_{data['num_simulations']:.0f}sim_"
                 f"{data['Npts']:.0e}np_{data['dt']:.0e}dt_"
                 f"{data['oversampling']}os_{data['prerun']:.0e}pre")
    
    if prefix is not None:
        base_name = f"{prefix}_{base_name}"
    
    ext = '.pickle'
    
    # Check if file already exists and handle accordingly
    dataset_name = base_name + ext
    dataset_path = os.path.join(folder_path, dataset_name)
    
    if merge and os.path.exists(dataset_path):
        # Check if the existing file has the same specs
        existing_specs = extract_specs(dataset_path)
        if data['num_simulations'] == existing_specs['num_simulations'] and \
           data['Npts'] == existing_specs['Npts'] and \
           data['dt'] == existing_specs['dt'] and \
           data['oversampling'] == existing_specs['oversampling'] and \
           data['prerun'] == existing_specs['prerun'] and \
           torch.equal(data['low_tensor'], existing_specs['low_tensor']) and \
           torch.equal(data['high_tensor'], existing_specs['high_tensor']):
            print(f"Overwriting existing dataset at {dataset_path}\n")
        else:
            # If specs don't match, add a counter to the filename
            counter = 1
            while os.path.exists(dataset_path):
                dataset_name = f'{base_name}_{counter}{ext}'
                dataset_path = os.path.join(folder_path, dataset_name)
                counter += 1
    else:
        # Check if file already exists and create a unique filename if it does
        counter = 1
        while os.path.exists(dataset_path):
            dataset_name = f'{base_name}_{counter}{ext}'
            dataset_path = os.path.join(folder_path, dataset_name)
            counter += 1
    
    # Save dataset
    with open(dataset_path, 'wb') as file:
        pickle.dump(data, file)
    
    print(f'Saved dataset at {dataset_path}\n')
    return dataset_path

def filter_files_by_specs(directory_path, target_Npts, target_dt, target_oversampling, target_prerun, target_low_tensor, target_high_tensor):
    """
    Filter files that have the same specs except for num_simulations and are not already merged.
    
    Parameters:
    directory_path (str): The path to the directory containing the dataset files.
    target_Npts (float): The target Npts specification.
    target_dt (float): The target dt specification.
    target_oversampling (int): The target oversampling specification.
    target_prerun (float): The target prerun specification.
    target_low_tensor (torch.Tensor): The target low_tensor specification.
    target_high_tensor (torch.Tensor): The target high_tensor specification.
    
    Returns:
    list: A list of filenames that match the specified criteria.
    """
    all_files = [file for file in os.listdir(directory_path) if file.endswith(".pickle")]

    def specs_match(filename):
        if filename.startswith("merged_"):
            return False
        try:
            specs = extract_specs(os.path.join(directory_path, filename))
            return (specs['Npts'] == target_Npts and
                    specs['dt'] == target_dt and
                    specs['oversampling'] == target_oversampling and
                    specs['prerun'] == target_prerun and
                    torch.equal(specs['low_tensor'], target_low_tensor) and
                    torch.equal(specs['high_tensor'], target_high_tensor))
        except (IndexError, ValueError):
            return False

    return [file for file in all_files if specs_match(file)]

def merge_datasets(directory_path, target_Npts, target_dt, target_oversampling, target_prerun, target_low_tensor, target_high_tensor, filter_half=None):
    """
    Merge datasets with the same specifications into one, with options to filter out half of the data.
    
    Parameters:
    directory_path (str): The path to the directory containing the dataset files.
    target_Npts (float): The target Npts specification.
    target_dt (float): The target dt specification.
    target_oversampling (int): The target oversampling specification.
    target_prerun (float): The target prerun specification.
    target_low_tensor (torch.Tensor): The target low_tensor specification.
    target_high_tensor (torch.Tensor): The target high_tensor specification.
    filter_half (str or None): 'first' to keep the first half, 'second' to keep the second half, or None to keep all data.
    
    Returns:
    dict: The merged dataset with combined data and updated metadata.
    """
    dataset_files = filter_files_by_specs(directory_path, target_Npts, target_dt, target_oversampling, target_prerun, target_low_tensor, target_high_tensor)

    if not dataset_files:
        raise FileNotFoundError("No files found with the target specifications for merging.")

    # Initialize empty tensors for theta and x data
    all_data_theta = torch.tensor([])
    all_data_x = torch.tensor([])

    for filename in dataset_files:
        file_path = os.path.join(directory_path, filename)
        data_x, data_theta = load_inference_data(file_path)

        all_data_theta = torch.cat((all_data_theta, data_theta))
        all_data_x = torch.cat((all_data_x, data_x))


    half_size = all_data_x.shape[1] // 2
    
    # Apply filter if specified
    if filter_half == 'corr':
        all_data_x = all_data_x[:, :half_size]
        prefix = 'merged_corr'
    elif filter_half == 'svr':
        all_data_x = all_data_x[:, half_size:]
        prefix = 'merged_svr'
    else:
        prefix = 'merged_full'
        filter_half = 'full'

    # Create the merged data dictionary
    merged_data = {
        'theta': all_data_theta,
        'x': all_data_x,
        'num_simulations': all_data_theta.shape[0],
        'Npts': target_Npts,
        'dt': target_dt,
        'oversampling': target_oversampling,
        'prerun': target_prerun,
        'low_tensor': target_low_tensor,
        'high_tensor': target_high_tensor,
        'data_type': filter_half,
        'prefix': prefix
    }

    # Save the merged dataset
    save_path = save_pickle_data(merged_data, directory_path, prefix = prefix, merge = True)
    #print("Dimensioni del dataset unificato:", all_data_theta.shape, all_data_x.shape)
    
    return save_path

def preprocess(data, features, density = 1.):
    """
    Extract specified features from a 3D numpy array of data and optionally downsample the data.

    Parameters:
    data (numpy.ndarray): A 3D array of shape (num_samples, num_features, num_points).
    features (list of str): A list of feature names to extract.
    density (float): A downsampling factor. 1.0 means no downsampling.

    Returns:
    numpy.ndarray: A 2D array of shape (num_samples, selected_features * downsampled_points)
                   containing the concatenated and downsampled data for the selected features.
    """
    # Selecting the wanted features
    string_to_int_map = {
        "Cxx": 0,
        "Cyy": 1,
        "S_red_x": 2,
        "S_red_y": 3,
    }
    step = int(1 / density)
    
    # Convert feature names to indices
    integer_list = [string_to_int_map[string] for string in features]
    
    # Collect arrays to be concatenated
    arrays_to_concatenate = [data[:, element, :] for element in integer_list]
    
    # Concatenate all selected arrays along the second axis
    new_data = torch.concatenate(arrays_to_concatenate, axis=1)

    step = int(1 / density)
    new_data = new_data[:,::step]
    #reducing the number of points

    return new_data