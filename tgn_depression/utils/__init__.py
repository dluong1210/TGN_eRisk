try:
    from .data_structures import (
        Conversation, 
        UserData, 
        UserBatch, 
        DepressionDataset,
        collate_users
    )
    from .data_loader import load_depression_data, create_dummy_data
    from .neighbor_finder import NeighborFinder, get_neighbor_finder, ConversationNeighborFinder
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
    from data_loader import load_depression_data, create_dummy_data
    from neighbor_finder import NeighborFinder, get_neighbor_finder, ConversationNeighborFinder
    from utils import (
        EarlyStopMonitor,
        MergeLayer,
        set_seed,
        get_device,
        compute_class_weights
    )
