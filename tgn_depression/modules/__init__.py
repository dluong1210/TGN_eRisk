try:
    from .memory import Memory
    from .message_function import MessageFunction, IdentityMessageFunction, MLPMessageFunction, get_message_function
    from .message_aggregator import MessageAggregator, LastMessageAggregator, MeanMessageAggregator, get_message_aggregator
    from .memory_updater import MemoryUpdater, GRUMemoryUpdater, RNNMemoryUpdater, get_memory_updater
    from .embedding_module import (
        EmbeddingModule, 
        IdentityEmbedding, 
        GraphAttentionEmbedding,
        GraphSumEmbedding,
        get_embedding_module
    )
except ImportError:
    from memory import Memory
    from message_function import MessageFunction, IdentityMessageFunction, MLPMessageFunction, get_message_function
    from message_aggregator import MessageAggregator, LastMessageAggregator, MeanMessageAggregator, get_message_aggregator
    from memory_updater import MemoryUpdater, GRUMemoryUpdater, RNNMemoryUpdater, get_memory_updater
    from embedding_module import (
        EmbeddingModule, 
        IdentityEmbedding, 
        GraphAttentionEmbedding,
        GraphSumEmbedding,
        get_embedding_module
    )
