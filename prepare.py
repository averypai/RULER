import json
import random

def split_dataset(data, train_ratio=0.96, val_ratio=0.0, test_ratio=0.04, random_seed=42):
    """
    Split a dataset into training, validation, and test sets using random indexing.
    
    Parameters:
    data: list or dict - The dataset to split
    train_ratio: float - Proportion of data for training (default: 0.6)
    val_ratio: float - Proportion of data for validation (default: 0.2)
    test_ratio: float - Proportion of data for testing (default: 0.2)
    random_seed: int - Random seed for reproducibility (default: 42)
    
    Returns:
    tuple: (train_set, val_set, test_set)
    """
    if not isinstance(data, (list, dict)):
        raise TypeError("Data must be a list or dictionary")
    
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-10:
        raise ValueError("Ratios must sum to 1")
    
    # Set random seed for reproducibility
    random.seed(random_seed)
    
    # Convert dict to list if necessary
    if isinstance(data, dict):
        items = list(data.items())
    else:
        items = data.copy()
    
    # Generate all indices
    total_size = len(items)
    all_indices = list(range(total_size))
    
    # Calculate split sizes
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    
    # Randomly select indices for each split
    train_indices = random.sample(all_indices, train_size)
    remaining_indices = list(set(all_indices) - set(train_indices))
    val_indices = random.sample(remaining_indices, val_size)
    test_indices = list(set(remaining_indices) - set(val_indices))
    
    # Create splits using the indices
    train_set = [items[i] for i in train_indices]
    val_set = [items[i] for i in val_indices]
    test_set = [items[i] for i in test_indices]
    
    # Convert back to dict if input was dict
    if isinstance(data, dict):
        train_set = dict(train_set)
        val_set = dict(val_set)
        test_set = dict(test_set)
    
    # Store indices used for each split
    split_indices = {
        'train_indices': train_indices,
        'val_indices': val_indices,
        'test_indices': test_indices
    }
    
    return train_set, val_set, test_set, split_indices

def organize_conclusion(dataset, filename):
    with open(filename, 'w') as f:
        for entry in dataset:
            conclusion = ''
            for c in dataset[entry]['CONTEXTS']:
                conclusion += f"{c}\n"
            conclusion += f"{dataset[entry]['LONG_ANSWER']}\n"

            data = {
            "idx": entry,
            'QUESTION': dataset[entry]['QUESTION'],
            'CONCLUSION': conclusion,
            "TYPE": dataset[entry]["MESHES"],
            "FINAL_DECISION": dataset[entry]["final_decision"]
            }
            f.write(json.dumps(data) + '\n')



# Example usage:
if __name__ == "__main__":
    # Load data
    with open('test_set.json', 'r') as file:
        data = json.load(file)

    # Split the dataset
    train_set, val_set, test_set, split_indices = split_dataset(data)

    # Print the sizes of each split
    print(f"Total dataset size: {len(data)}")
    print(f"Training set size: {len(train_set)}")
    print(f"Validation set size: {len(val_set)}")
    print(f"Test set size: {len(test_set)}")

    # Save the splits and indices
    with open('data/train_set.json', 'w') as f:
        json.dump(train_set, f, indent=2)
        organize_conclusion(train_set, 'haystack.jsonl')

    with open('data/val_set.json', 'w') as f:
        json.dump(val_set, f, indent=2)
        
    with open('data/test_set.json', 'w') as f:
        json.dump(test_set, f, indent=2)
        organize_conclusion(test_set, 'needle.jsonl')
        
    # Save the indices used for splitting
    with open('data/split_indices.json', 'w') as f:
        json.dump(split_indices, f, indent=2)