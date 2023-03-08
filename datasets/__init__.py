from datasets.H2gigaDataset import H2gigaDataset

def get_dataset(name, dataset_opts):
    if name == "H2giga": 
        return H2gigaDataset(**dataset_opts)
    else:
        raise RuntimeError("Dataset {} not available".format(name))