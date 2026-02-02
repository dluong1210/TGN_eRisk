# TGN for Depression Detection

Temporal Graph Networks adapted for depression detection from Reddit conversations.

## Problem Setup

- **Input**: Mỗi TARGET USER có NHIỀU conversations theo thời gian
- **Output**: Binary classification - user có depression hay không
- **Label**: MỘT label duy nhất cho CẢ user (không phải per-conversation)

**Target users là ĐỘC LẬP**: Mỗi target_user có chuỗi conversations riêng, trong đó họ tương tác với các user khác trên social (không phải target_user tương tác với nhau). Train/val/test split là **random stratified** (giữ tỉ lệ label), không chronological.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     TGN Depression Detection                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  Input: Target User với tất cả conversations                      │
│  - Mỗi conversation là temporal graph                             │
│  - Nodes: Users tham gia                                          │
│  - Edges: Posts/Replies (với embeddings)                          │
│                                                                   │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │                    TGN ENCODER                              │  │
│  │                                                              │  │
│  │  1. Memory Module      - State vector for each user         │  │
│  │  2. Message Function   - Creates messages from events       │  │
│  │  3. Message Aggregator - Aggregates messages (last/mean)    │  │
│  │  4. Memory Updater     - GRU updates memory                 │  │
│  │  5. Embedding Module   - Temporal graph attention           │  │
│  │                                                              │  │
│  └────────────────────────────────────────────────────────────┘  │
│                              │                                    │
│                              ▼                                    │
│               Process ALL conversations của user                  │
│               (theo thứ tự thời gian trong chuỗi của user đó)     │
│                              │                                    │
│                              ▼                                    │
│                    Final User Embedding z_i(t)                    │
│                              │                                    │
│                              ▼                                    │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │                CLASSIFICATION HEAD                          │  │
│  │                MLP(z_i) → P(depression)                     │  │
│  └────────────────────────────────────────────────────────────┘  │
│                                                                   │
│  Output: Depression probability                                   │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

## Data Format

### 1. Interactions CSV (`interactions.csv`)

```csv
userID,parentID,timestamp,post_id,conversation_id
user_123,user_456,1609459200,post_1,conv_1
user_456,user_123,1609459500,post_2,conv_1
user_789,user_123,1609550000,post_3,conv_2
...
```

- **userID**: User gửi post/reply
- **parentID**: User được reply (parent của post này). NaN cho root posts.
- **timestamp**: Thời gian của post
- **post_id**: ID của post
- **conversation_id**: ID của conversation

### 2. Embeddings JSON (`embeddings.json`)

Pre-computed BERT embeddings của posts, organized by target user.

```json
{
    "target_user_1": {
        "post_1": [0.1, 0.2, ...],
        "post_3": [0.3, 0.4, ...]
    },
    "target_user_2": {
        "post_2": [0.5, 0.6, ...],
        "post_4": [0.7, 0.8, ...]
    }
}
```

### 3. Labels JSON (`labels.json`)

**QUAN TRỌNG**: Mỗi user CHỈ CÓ MỘT label

```json
{
    "target_user_1": 1,
    "target_user_2": 0,
    "target_user_3": 1
}
```

- 0 = không depression
- 1 = depression

## Usage

### Training with Dummy Data (Testing)

```bash
python train.py --use_dummy_data --epochs 10
```

### Training with Real Data

Đặt các file vào thư mục data:
- `data/interactions.csv`
- `data/embeddings.json`
- `data/labels.json`

Split mặc định: `--split_method stratified` (giữ tỉ lệ depression/non-depression trong train/val/test). Có thể dùng `--split_method random`.

```bash
python train.py \
    --data_dir ./data \
    --epochs 50 \
    --n_layers 1 \
    --n_heads 2 \
    --memory_dim 768 \
    --n_neighbors 10 \
    --lr 0.0001 \
    --patience 5
```

### Command Line Arguments

**Data:**
- `--data_dir`: Directory containing data files
- `--use_dummy_data`: Use generated dummy data
- `--embedding_dim`: Embedding dimension (default: 768)

**Model:**
- `--n_layers`: Number of graph attention layers (default: 1)
- `--n_heads`: Number of attention heads (default: 2)
- `--memory_dim`: Memory dimension (default: 172)
- `--n_neighbors`: Number of neighbors to sample (default: 10)
- `--embedding_module`: 'graph_attention', 'graph_sum', or 'identity'
- `--use_memory` / `--no_memory`: Enable/disable memory module

**Training:**
- `--epochs`: Number of epochs (default: 50)
- `--lr`: Learning rate (default: 0.0001)
- `--patience`: Early stopping patience (default: 5)

## Training Flow

For each TARGET USER:

```
1. Reset Memory → Initialize all user memories to zeros

2. Get ALL conversations of target user (sorted by time)

3. Get ALL interactions across ALL conversations (sorted by time)

4. Process interactions in batches:
   ┌────────────────────────────────────────────────────┐
   │ For each batch of interactions:                    │
   │   - Update memory from previous batch's messages   │
   │   - Store new messages for next batch              │
   └────────────────────────────────────────────────────┘

5. Compute target user embedding at final timestamp
   using temporal graph attention

6. Classify: MLP(embedding) → P(depression)

7. Loss + Backprop
```

## Example

```python
from model.tgn_depression import TGNDepression
from utils.neighbor_finder import UserNeighborFinder

# Initialize model
model = TGNDepression(
    n_users=1000,
    edge_features=post_embeddings,
    device=device,
    n_layers=1,
    use_memory=True,
    memory_dimension=768
)

# For each target user
for user_data in dataset:
    # Reset memory
    model.reset_memory()
    
    # Create neighbor finder from ALL user's conversations
    neighbor_finder = UserNeighborFinder(user_data, n_users=1000)
    model.set_neighbor_finder(neighbor_finder)
    
    # Forward pass - processes ALL conversations
    logits = model.forward_user(user_data)
    
    # Get prediction
    prob_depression = torch.softmax(logits, dim=1)[0, 1]
```

## Project Structure

```
tgn_depression/
├── model/
│   └── tgn_depression.py      # Main TGN model
├── modules/
│   ├── memory.py              # Memory module
│   ├── message_function.py    # Message creation
│   ├── message_aggregator.py  # Message aggregation
│   ├── memory_updater.py      # GRU/RNN memory update
│   └── embedding_module.py    # Graph attention embedding
├── utils/
│   ├── data_structures.py     # UserData, DepressionDataset
│   ├── data_loader.py         # Data loading
│   ├── neighbor_finder.py     # Temporal neighbor sampling
│   └── utils.py               # Helper functions
├── train.py                   # Training script
├── test_model.py              # Test script
└── README.md
```

## Key Classes

### UserData
```python
@dataclass
class UserData:
    user_id: int                  # User index
    user_id_str: str              # Original user ID
    conversations: List[Conversation]  # ALL conversations
    label: int                    # Depression label (0 or 1)
```

### Conversation
```python
@dataclass  
class Conversation:
    conversation_id: str
    source_users: np.ndarray      # Users who sent posts
    dest_users: np.ndarray        # Users who received replies
    timestamps: np.ndarray        # Timestamps
    post_ids: np.ndarray          # Post IDs (for embeddings)
```

## References

- [Temporal Graph Networks Paper](https://arxiv.org/abs/2006.10637)
