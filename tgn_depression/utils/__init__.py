try:
    from .data_structures import (
        Conversation, 
        UserData, 
        UserBatch, 
        DepressionDataset,
        collate_users
    )
    from .data_loader import load_depression_data, load_depression_data_from_parquet_folders, create_dummy_data
    from .neighbor_finder import NeighborFinder, get_neighbor_finder
    from .utils import (
        EarlyStopMonitor,
        MergeLayer,
        set_seed,
        get_device,
        compute_class_weights
    )
except ImportError:
    from data_structures import (
        Conversation, 
        UserData, 
        UserBatch, 
        DepressionDataset,
        collate_users
    )
    from data_loader import load_depression_data, load_depression_data_from_parquet_folders, create_dummy_data
    from neighbor_finder import NeighborFinder, get_neighbor_finder
    from utils import (
        EarlyStopMonitor,
        MergeLayer,
        set_seed,
        get_device,
        compute_class_weights
    )
