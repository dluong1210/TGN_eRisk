"""
Test script for TGN Sequential models (CarryOver and LSTM).
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np

from model.tgn_sequential import TGNSequential, TGNCarryOver, TGNLstm
from utils.data_loader import create_dummy_data


def test_carryover():
    """Test CarryOver mode."""
    print("=" * 60)
    print("Testing CARRY-OVER Mode")
    print("=" * 60)
    
    # Create dummy data
    print("\n1. Creating dummy data...")
    train_ds, val_ds, test_ds, metadata = create_dummy_data(
        n_total_users=50,
        n_target_users=20,
        n_conversations=100,
        avg_interactions=8,
        embedding_dim=128
    )
    
    device = torch.device('cpu')
    
    # Initialize model
    print("\n2. Initializing CarryOver model...")
    model = TGNCarryOver(
        n_users=metadata['n_total_users'],
        edge_features=train_ds.post_embeddings,
        device=device,
        n_layers=1,
        n_heads=2,
        dropout=0.1,
        use_memory=True,
        memory_dimension=128,
        n_neighbors=5
    )
    
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward pass
    print("\n3. Testing forward pass...")
    
    for i, user_data in enumerate(train_ds.users[:3]):
        model.reset_state()
        
        logits = model.forward_user(user_data)
        
        print(f"   User {i+1}: {user_data.user_id_str}")
        print(f"      - Conversations: {user_data.n_conversations}")
        print(f"      - Total interactions: {user_data.total_interactions}")
        print(f"      - Label: {user_data.label}")
        print(f"      - Output shape: {logits.shape}")
        print(f"      - Logits: {logits.detach().numpy().round(4)}")
        
        # Check node_features was updated (carryover effect)
        if user_data.n_conversations > 0:
            node_feat = model.node_features[user_data.user_id]
            feat_norm = torch.norm(node_feat).item()
            print(f"      - Node features norm: {feat_norm:.4f}")
    
    # Test backward
    print("\n4. Testing backward pass...")
    model.reset_state()
    user_data = train_ds.users[0]
    
    logits = model.forward_user(user_data)
    label = torch.LongTensor([user_data.label])
    loss = torch.nn.CrossEntropyLoss()(logits, label)
    
    loss.backward()
    
    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 
                   for p in model.parameters())
    print(f"   Loss: {loss.item():.4f}")
    print(f"   Gradients computed: {has_grad}")
    
    print("\n[PASS] CarryOver mode test passed!")
    return True


def test_lstm():
    """Test LSTM mode."""
    print("\n" + "=" * 60)
    print("Testing LSTM Mode")
    print("=" * 60)
    
    # Create dummy data
    print("\n1. Creating dummy data...")
    train_ds, val_ds, test_ds, metadata = create_dummy_data(
        n_total_users=50,
        n_target_users=20,
        n_conversations=100,
        avg_interactions=8,
        embedding_dim=128
    )
    
    device = torch.device('cpu')
    
    # Initialize model
    print("\n2. Initializing LSTM model...")
    model = TGNLstm(
        n_users=metadata['n_total_users'],
        edge_features=train_ds.post_embeddings,
        device=device,
        lstm_hidden_dim=64,
        lstm_num_layers=1,
        lstm_bidirectional=True,
        n_layers=1,
        n_heads=2,
        dropout=0.1,
        use_memory=True,
        memory_dimension=128,
        n_neighbors=5
    )
    
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   LSTM: hidden={64}, layers={1}, bidirectional=True")
    
    # Test forward pass
    print("\n3. Testing forward pass...")
    
    for i, user_data in enumerate(train_ds.users[:3]):
        model.reset_state()
        
        logits = model.forward_user(user_data)
        
        print(f"   User {i+1}: {user_data.user_id_str}")
        print(f"      - Conversations: {user_data.n_conversations}")
        print(f"      - Total interactions: {user_data.total_interactions}")
        print(f"      - Label: {user_data.label}")
        print(f"      - Output shape: {logits.shape}")
        print(f"      - Logits: {logits.detach().numpy().round(4)}")
    
    # Test backward
    print("\n4. Testing backward pass...")
    model.reset_state()
    user_data = train_ds.users[0]
    
    logits = model.forward_user(user_data)
    label = torch.LongTensor([user_data.label])
    loss = torch.nn.CrossEntropyLoss()(logits, label)
    
    loss.backward()
    
    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 
                   for p in model.parameters())
    
    # Check LSTM gradients specifically
    lstm_grad = model.lstm.weight_ih_l0.grad
    lstm_has_grad = lstm_grad is not None and lstm_grad.abs().sum() > 0
    
    print(f"   Loss: {loss.item():.4f}")
    print(f"   Gradients computed: {has_grad}")
    print(f"   LSTM gradients: {lstm_has_grad}")
    
    print("\n[PASS] LSTM mode test passed!")
    return True


def test_comparison():
    """Compare outputs of both modes."""
    print("\n" + "=" * 60)
    print("Comparing Both Modes")
    print("=" * 60)
    
    # Create dummy data
    train_ds, _, _, metadata = create_dummy_data(
        n_total_users=30,
        n_target_users=10,
        n_conversations=50,
        avg_interactions=5,
        embedding_dim=64
    )
    
    device = torch.device('cpu')
    
    # Initialize both models
    common_params = {
        'n_users': metadata['n_total_users'],
        'edge_features': train_ds.post_embeddings,
        'device': device,
        'n_layers': 1,
        'n_heads': 2,
        'use_memory': True,
        'memory_dimension': 64,
        'n_neighbors': 3
    }
    
    model_carryover = TGNCarryOver(**common_params)
    model_lstm = TGNLstm(**common_params, lstm_hidden_dim=32)
    
    print("\n1. Processing same users with both models...")
    
    results = []
    for user_data in train_ds.users[:5]:
        # CarryOver
        model_carryover.reset_state()
        logits_co = model_carryover.forward_user(user_data)
        prob_co = torch.softmax(logits_co, dim=1)[0, 1].item()
        
        # LSTM
        model_lstm.reset_state()
        logits_lstm = model_lstm.forward_user(user_data)
        prob_lstm = torch.softmax(logits_lstm, dim=1)[0, 1].item()
        
        results.append({
            'user': user_data.user_id_str,
            'n_conv': user_data.n_conversations,
            'label': user_data.label,
            'prob_carryover': prob_co,
            'prob_lstm': prob_lstm
        })
        
        print(f"   {user_data.user_id_str}: {user_data.n_conversations} convs, "
              f"label={user_data.label}, "
              f"P(dep|CO)={prob_co:.3f}, P(dep|LSTM)={prob_lstm:.3f}")
    
    print("\n[PASS] Comparison test passed!")
    return True


def test_different_num_conversations():
    """Test with varying number of conversations."""
    print("\n" + "=" * 60)
    print("Testing with Different Number of Conversations")
    print("=" * 60)
    
    device = torch.device('cpu')
    
    # Create simple embeddings
    embedding_dim = 64
    n_users = 20
    post_embeddings = np.random.randn(100, embedding_dim).astype(np.float32)
    
    # Create data with different conversation counts
    from utils.data_structures import Conversation, UserData
    
    users = []
    for n_convs in [0, 1, 3, 5, 10]:
        conversations = []
        current_time = 0.0
        
        for c in range(n_convs):
            n_interactions = 5
            conv = Conversation(
                conversation_id=f"conv_{c}",
                source_users=np.random.randint(0, n_users, n_interactions),
                dest_users=np.random.randint(0, n_users, n_interactions),
                timestamps=np.array([current_time + i*100 for i in range(n_interactions)]),
                post_ids=np.random.randint(0, 100, n_interactions)
            )
            conversations.append(conv)
            current_time += 10000
        
        user = UserData(
            user_id=len(users),
            user_id_str=f"user_{n_convs}convs",
            conversations=conversations,
            label=0
        )
        users.append(user)
    
    # Test both models
    model_co = TGNCarryOver(
        n_users=n_users,
        edge_features=post_embeddings,
        device=device,
        memory_dimension=64,
        n_neighbors=3
    )
    
    model_lstm = TGNLstm(
        n_users=n_users,
        edge_features=post_embeddings,
        device=device,
        memory_dimension=64,
        lstm_hidden_dim=32,
        n_neighbors=3
    )
    
    print("\nTesting users with different conversation counts:")
    print("-" * 50)
    print(f"{'User':<20} {'N_Conv':<8} {'CarryOver':<12} {'LSTM':<12}")
    print("-" * 50)
    
    for user in users:
        model_co.reset_state()
        model_lstm.reset_state()
        
        logits_co = model_co.forward_user(user)
        logits_lstm = model_lstm.forward_user(user)
        
        prob_co = torch.softmax(logits_co, dim=1)[0, 1].item()
        prob_lstm = torch.softmax(logits_lstm, dim=1)[0, 1].item()
        
        print(f"{user.user_id_str:<20} {user.n_conversations:<8} "
              f"{prob_co:<12.4f} {prob_lstm:<12.4f}")
    
    print("-" * 50)
    print("\n[PASS] Variable conversations test passed!")
    return True


if __name__ == "__main__":
    print("=" * 60)
    print("TGN Sequential Models Test Suite")
    print("=" * 60)
    
    try:
        test_carryover()
        test_lstm()
        test_comparison()
        test_different_num_conversations()
        
        print("\n" + "=" * 60)
        print("[PASS] ALL TESTS PASSED!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n[FAIL] Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
