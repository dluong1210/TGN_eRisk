"""
Quick test script to verify the TGN Depression model works.

Run: python test_model.py
"""

import numpy as np
import torch
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model.tgn_depression import TGNDepression
from utils.data_structures import UserData
from utils.data_loader import create_dummy_data
from utils.neighbor_finder import UserNeighborFinder
from utils.utils import set_seed, get_device


def test_basic():
    """Test basic model forward pass."""
    print("=" * 50)
    print("Testing TGN Depression Model")
    print("=" * 50)
    
    set_seed(42)
    device = get_device(0)
    print(f"Device: {device}")
    
    # Create dummy data
    print("\n1. Creating dummy data...")
    train_dataset, val_dataset, test_dataset, metadata = create_dummy_data(
        n_total_users=100,
        n_target_users=50,
        n_conversations=100,
        avg_interactions=8,
        embedding_dim=128,
        depression_ratio=0.3
    )
    
    print(f"   - {metadata['n_total_users']} total users")
    print(f"   - {metadata['n_target_users']} target users")
    print(f"   - {metadata['n_posts']} posts")
    print(f"   - {len(train_dataset)} train users")
    print(f"   - {len(val_dataset)} val users")
    print(f"   - {len(test_dataset)} test users")
    
    # Initialize model
    print("\n2. Initializing model...")
    model = TGNDepression(
        n_users=metadata['n_total_users'],
        edge_features=train_dataset.post_embeddings,
        device=device,
        n_layers=1,
        n_heads=2,
        dropout=0.1,
        use_memory=True,
        memory_dimension=128,  # Match embedding_dim
        embedding_module_type='graph_attention',
        n_neighbors=5
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"   - Model parameters: {n_params:,}")
    
    # Test forward pass
    print("\n3. Testing forward pass...")
    user_data = train_dataset[0]
    print(f"   - User: {user_data.user_id_str}")
    print(f"   - Conversations: {user_data.n_conversations}")
    print(f"   - Total interactions: {user_data.total_interactions}")
    print(f"   - Label: {user_data.label}")
    
    # Reset memory
    model.reset_memory()
    
    # Create neighbor finder from user's data
    neighbor_finder = UserNeighborFinder(
        user_data=user_data,
        n_users=metadata['n_total_users'],
        uniform=False
    )
    model.set_neighbor_finder(neighbor_finder)
    
    # Forward pass
    logits = model.forward_user(user_data, n_neighbors=5)
    
    print(f"   - Output logits shape: {logits.shape}")
    print(f"   - Logits: {logits.detach().cpu().numpy()}")
    
    probs = torch.softmax(logits, dim=1)
    print(f"   - Probabilities: {probs.detach().cpu().numpy()}")
    
    # Test backward pass
    print("\n4. Testing backward pass...")
    criterion = torch.nn.CrossEntropyLoss()
    label = torch.LongTensor([user_data.label]).to(device)
    loss = criterion(logits, label)
    print(f"   - Loss: {loss.item():.4f}")
    
    loss.backward()
    print("   - Backward pass successful!")
    
    # Check gradients
    has_grad = False
    for name, param in model.named_parameters():
        if param.grad is not None and param.grad.abs().sum() > 0:
            has_grad = True
            break
    
    print(f"   - Gradients computed: {has_grad}")
    
    # Test multiple users
    print("\n5. Testing multiple users...")
    total_loss = 0
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    for i, user_data in enumerate(train_dataset.users[:5]):
        model.reset_memory()
        
        neighbor_finder = UserNeighborFinder(
            user_data=user_data,
            n_users=metadata['n_total_users']
        )
        model.set_neighbor_finder(neighbor_finder)
        
        optimizer.zero_grad()
        
        logits = model.forward_user(user_data, n_neighbors=5)
        
        label = torch.LongTensor([user_data.label]).to(device)
        loss = criterion(logits, label)
        loss.backward()
        optimizer.step()
        
        model.detach_memory()
        
        total_loss += loss.item()
        print(f"   - User {i+1} ({user_data.n_conversations} convs, {user_data.total_interactions} interactions): Loss = {loss.item():.4f}")
    
    avg_loss = total_loss / 5
    print(f"   - Average loss: {avg_loss:.4f}")
    
    print("\n" + "=" * 50)
    print("All tests passed!")
    print("=" * 50)
    
    return True


def test_no_memory():
    """Test model without memory module."""
    print("\n" + "=" * 50)
    print("Testing TGN without Memory")
    print("=" * 50)
    
    set_seed(42)
    device = get_device(0)
    
    # Create data
    train_dataset, _, _, metadata = create_dummy_data(
        n_total_users=50,
        n_target_users=25,
        n_conversations=50,
        avg_interactions=5,
        embedding_dim=128
    )
    
    # Model without memory
    model = TGNDepression(
        n_users=metadata['n_total_users'],
        edge_features=train_dataset.post_embeddings,
        device=device,
        n_layers=1,
        use_memory=False,
        embedding_module_type='graph_sum'
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test
    user_data = train_dataset[0]
    model.reset_memory()
    
    neighbor_finder = UserNeighborFinder(
        user_data=user_data,
        n_users=metadata['n_total_users']
    )
    model.set_neighbor_finder(neighbor_finder)
    
    logits = model.forward_user(user_data, n_neighbors=5)
    
    print(f"Output shape: {logits.shape}")
    print("No memory test passed!")
    
    return True


if __name__ == "__main__":
    try:
        test_basic()
        test_no_memory()
        print("\n[PASS] All tests passed successfully!")
    except Exception as e:
        print(f"\n[FAIL] Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
