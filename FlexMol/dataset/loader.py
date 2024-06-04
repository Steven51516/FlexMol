import numpy as np
import pandas as pd


def load_DDI(source, from_df=False, drug1="Drug1", drug2="Drug2", label="Y", delimiter=",", split_frac = None, shuffle = True):
    """
    Load DDI data either from a file or a DataFrame.
    """
    if not from_df:
        df = pd.read_csv(source, delimiter=delimiter)
    else:
        df = source

    df_selected = df[[drug1, drug2, label]].rename(columns={drug1: 'Drug1', drug2: 'Drug2', label: 'Y'})

    if(split_frac):
        return split(df_selected, split_frac, shuffle)
    
    return df_selected



def load_DTI(source, from_df=False, drug="Drug", target="Protein", label="Y", delimiter=",", id = True, protein_id="Protein_ID", split_frac = None, shuffle = True):
    """
    Load DTI data either from a file or a DataFrame.
    """
    if not from_df:
        df = pd.read_csv(source, delimiter=delimiter)
    else:
        df = source

    columns = [drug, target, label]
    rename_dict = {drug: 'Drug', target: 'Protein', label: 'Y'}
    
    if id:
        columns.append(protein_id)
        rename_dict[protein_id] = 'Protein_ID'
    
    df_selected = df[columns].rename(columns=rename_dict)

    if(split_frac):
        return split(df_selected, split_frac, shuffle)
    
    return df_selected



def load_PPI(source, from_df=False, protein1="Protein1", protein2="Protein2", label="Y", delimiter=",", id = True, protein1_id="Protein1_ID", protein2_id="Protein2_ID", split_frac = None, shuffle = True):
    """
    Load PPI data either from a file or a DataFrame.
    """
    if not from_df:
        df = pd.read_csv(source, delimiter=delimiter)
    else:
        df = source
    columns = [protein1, protein2, label]
    rename_dict = {protein1: 'Protein1', protein2: 'Protein2', label: 'Y'}
    
    if id:
        columns.append(protein1_id)
        rename_dict[protein1_id] = 'Protein1_ID'
        columns.append(protein2_id)
        rename_dict[protein2_id] = 'Protein2_ID'
    
    df_selected = df[columns].rename(columns=rename_dict)

    if(split_frac):
        return split(df_selected, split_frac, shuffle)
    
    return df_selected


def load_helper(path):
    result = []
    with open(path, 'r') as file:
        for line in file:
            line = line.strip()
            elements = line.split()
            result.append(elements)
    data_transposed = list(map(list, zip(*result)))
    return data_transposed


def load_BIOSNAP(file_path):
    df = pd.read_csv(file_path)
    df = df[df['SMILES'].apply(lambda x: len(x) <= 512)]
    columns_mapping = {
        'Gene': 'Protein_ID',
        'SMILES': 'Drug',
        'Target Sequence': 'Protein',
        'Label': 'Y'
    }
    df_renamed = df[list(columns_mapping.keys())].rename(columns=columns_mapping)
    return df_renamed


def load_DAVIS(path):
    data = load_helper(path)
    drugs = data[1]
    targets = data[2]
    affinity = data[3]
    ids = data[0]
    affinity = [float(i) for i in affinity]
    df_data = {
        'Drug': targets,
        'Protein': drugs,
        'Y': affinity, 
        "Protein_ID": ids

    }
    df = pd.DataFrame(df_data)
    return df


def split(df, split_frac, shuffle=True):
    """
    Splits the DataFrame into multiple DataFrames based on the given fractions.

    Args:
        df: The DataFrame to split.
        split_frac: List of fractions that sum to 1.
        shuffle: Whether to shuffle the DataFrame before splitting.

    Returns:
        List of DataFrames.
    """
    assert len(split_frac) > 1, "Fraction list must contain at least two values."
    assert np.isclose(sum(split_frac), 1.0), "Fractions must sum to 1."

    if shuffle:
        df = df.sample(frac=1).reset_index(drop=True)

    data_splits = []
    cumulative_frac = np.cumsum(split_frac)
    split_indices = (cumulative_frac * len(df)).astype(int)

    prev_idx = 0
    for idx in split_indices[:-1]:
        data_splits.append(df.iloc[prev_idx:idx].reset_index(drop=True))
        prev_idx = idx

    data_splits.append(df.iloc[prev_idx:].reset_index(drop=True))
    return tuple(data_splits)