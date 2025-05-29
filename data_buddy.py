import pickle
import os


def get_link_prediction_data_buddy(data_name, train_ratio, valid_ratio, K, seed=42, save_dir='.'):
    save_path = os.path.join(save_dir, f'{data_name}_{train_ratio}_{valid_ratio}_{K}_{seed}.pkl')
    if os.path.exists(save_path):
        # If the file exists, then open and load it
        with open(save_path, 'rb') as pickle_file:
            result_data = pickle.load(pickle_file)
        print("Successfully loaded file.")
        return result_data
    else:
        raise ValueError(f'{save_path} does not exist!')
    
    