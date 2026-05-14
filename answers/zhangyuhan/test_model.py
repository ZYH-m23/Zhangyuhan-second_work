from einops import rearrange
import numpy
import torch
import torch.nn.functional as F

from .adapters import (
    run_multihead_self_attention_with_rope,
    run_rope,
    run_silu,
    run_multihead_self_attention,
    run_swiglu,
    run_rmsnorm,
    run_scaled_dot_product_attention,
    run_transformer_block,
    run_transformer_lm,
    run_linear,
    run_embedding,
    run_abstopk,
    run_attention_with_sink,
    run_magnitude_pruning
)


def test_linear(numpy_snapshot, ts_state_dict, in_embeddings, d_model, d_ff):
    w1_weight = ts_state_dict[0]["layers.0.ffn.w1.weight"]
    output = run_linear(
        d_in=d_model,
        d_out=d_ff,
        weights=w1_weight,
        in_features=in_embeddings,
    )
    numpy_snapshot.assert_match(output)


def test_embedding(numpy_snapshot, ts_state_dict, in_indices, vocab_size, d_model):
    embedding_weight = ts_state_dict[0]["token_embeddings.weight"]
    output = run_embedding(
        vocab_size=vocab_size,
        d_model=d_model,
        weights=embedding_weight,
        token_ids=in_indices,
    )
    numpy_snapshot.assert_match(output)


def test_swiglu(numpy_snapshot, ts_state_dict, in_embeddings, d_model, d_ff):
    w1_weight, w2_weight, w3_weight = [ts_state_dict[0][f"layers.0.ffn.{k}.weight"] for k in ["w1", "w2", "w3"]]

    actual_output = run_swiglu(
        d_model=d_model,
        d_ff=d_ff,
        w1_weight=w1_weight,
        w2_weight=w2_weight,
        w3_weight=w3_weight,
        in_features=in_embeddings,
    )
    numpy_snapshot.assert_match(actual_output, atol=1e-5)


def test_scaled_dot_product_attention(numpy_snapshot, q, k, v, mask):
    actual_output = run_scaled_dot_product_attention(Q=q, K=k, V=v, mask=mask)
    numpy_snapshot.assert_match(
        actual_output,
        atol=1e-6,
    )


def test_4d_scaled_dot_product_attention(numpy_snapshot, q, k, v, mask):
    # Shape: (batch_size, num_heads, seq_len, d_k)
    q, k, v = (rearrange(x, "(batch head) seq d -> batch head seq d", head=2) for x in (q, k, v))
    mask = rearrange(mask, "(batch head) query key -> batch head query key", head=2)

    actual_output = run_scaled_dot_product_attention(Q=q, K=k, V=v, mask=mask)
    numpy_snapshot.assert_match(
        actual_output,
        atol=1e-6,
    )


def test_multihead_self_attention(numpy_snapshot, in_embeddings, d_model, n_heads, ts_state_dict):
    d, _ = ts_state_dict
    q_proj_weight, k_proj_weight, v_proj_weight, o_proj_weight = [
        d[f"layers.0.attn.{k}_proj.weight"] for k in ["q", "k", "v", "output"]
    ]
    actual_output = run_multihead_self_attention(
        d_model=d_model,
        num_heads=n_heads,
        q_proj_weight=q_proj_weight,
        k_proj_weight=k_proj_weight,
        v_proj_weight=v_proj_weight,
        o_proj_weight=o_proj_weight,
        in_features=in_embeddings,
    )
    numpy_snapshot.assert_match(actual_output, atol=1e-6)


def test_multihead_self_attention_with_rope(
    numpy_snapshot, in_embeddings, d_model, n_heads, ts_state_dict, n_keys, theta, pos_ids
):
    d, _ = ts_state_dict
    q_proj_weight, k_proj_weight, v_proj_weight, o_proj_weight = [
        d[f"layers.0.attn.{k}_proj.weight"] for k in ["q", "k", "v", "output"]
    ]
    pos_ids = rearrange(pos_ids, "seq -> 1 seq")
    actual_output = run_multihead_self_attention_with_rope(
        d_model=d_model,
        num_heads=n_heads,
        max_seq_len=n_keys,
        theta=theta,
        q_proj_weight=q_proj_weight,
        k_proj_weight=k_proj_weight,
        v_proj_weight=v_proj_weight,
        o_proj_weight=o_proj_weight,
        in_features=in_embeddings,
        token_positions=pos_ids,
    )
    numpy_snapshot.assert_match(actual_output, atol=1e-6)


def test_transformer_lm(
    numpy_snapshot, vocab_size, n_keys, d_model, n_layers, n_heads, d_ff, theta, ts_state_dict, in_indices
):
    state_dict, _ = ts_state_dict

    actual_output = run_transformer_lm(
        vocab_size=vocab_size,
        context_length=n_keys,
        d_model=d_model,
        num_layers=n_layers,
        num_heads=n_heads,
        d_ff=d_ff,
        rope_theta=theta,
        weights=state_dict,
        in_indices=in_indices,
    )
    numpy_snapshot.assert_match(actual_output, atol=1e-4, rtol=1e-2)


def test_transformer_lm_truncated_input(
    numpy_snapshot, vocab_size, n_keys, d_model, n_layers, n_heads, d_ff, theta, ts_state_dict, in_indices
):
    in_indices_truncated = in_indices[..., : in_indices.shape[-1] // 2]
    truncated_actual_output = run_transformer_lm(
        vocab_size=vocab_size,
        context_length=n_keys,
        d_model=d_model,
        num_layers=n_layers,
        num_heads=n_heads,
        d_ff=d_ff,
        rope_theta=theta,
        weights=ts_state_dict[0],
        in_indices=in_indices_truncated,
    )

    numpy_snapshot.assert_match(
        truncated_actual_output,
        atol=1e-4,
    )


def test_transformer_block(numpy_snapshot, ts_state_dict, in_embeddings, d_model, n_heads, d_ff, n_keys, theta):
    block_weights = {k.replace("layers.0.", ""): v for k, v in ts_state_dict[0].items() if "layers.0." in k}

    actual_output = run_transformer_block(
        d_model=d_model,
        num_heads=n_heads,
        d_ff=d_ff,
        max_seq_len=n_keys,
        theta=theta,
        weights=block_weights,
        in_features=in_embeddings,
    )
    numpy_snapshot.assert_match(
        actual_output,
        atol=1e-6,
    )


def test_rmsnorm(numpy_snapshot, ts_state_dict, in_embeddings):
    state_dict, _ = ts_state_dict
    reference_weights = state_dict["layers.1.ln1.weight"]
    d_model = reference_weights.shape[0]

    actual_output = run_rmsnorm(d_model=d_model, eps=1e-5, weights=reference_weights, in_features=in_embeddings)

    numpy_snapshot.assert_match(actual_output, atol=1e-6)


def test_rope(numpy_snapshot, in_embeddings, d_model, theta, n_queries, pos_ids):
    output = run_rope(
        d_model, theta=theta, max_seq_len=n_queries, in_query_or_key=in_embeddings, token_positions=pos_ids
    )
    numpy_snapshot.assert_match(output, atol=1e-6)


def test_silu_matches_pytorch():
    x = torch.tensor(
        [
            [0.2352, 0.9259, 0.5189, 0.4725, 0.9730],
            [0.7581, 0.9692, 0.2129, 0.9345, 0.0149],
        ]
    )
    expected_output = F.silu(x)
    actual_output = run_silu(x)
    numpy.testing.assert_allclose(actual_output.detach().numpy(), expected_output.detach().numpy(), atol=1e-6)

def test_abstopk_forward():
    """
    Test if AbsTopK correctly keeps the top-k absolute values and zeroes others.
    """
    # Create a deterministic tensor: shape (2, 5)
    # Row 0: [1, -5, 2, -4, 3] -> Abs: [1, 5, 2, 4, 3] -> Top 2 are 5 (-5) and 4 (-4)
    # Row 1: [0.1, 0.2, 0.3, 0.4, 0.5] -> Top 2 are 0.5 and 0.4
    x = torch.tensor([
        [1.0, -5.0, 2.0, -4.0, 3.0],
        [0.1, 0.2, 0.3, 0.4, 0.5]
    ])
    k = 2
    
    expected = torch.tensor([
        [0.0, -5.0, 0.0, -4.0, 0.0],
        [0.0, 0.0, 0.0, 0.4, 0.5]
    ])
    
    output = run_abstopk(x, k=k)
    
    torch.testing.assert_close(output, expected)
    
    # Verify sparsity
    assert torch.count_nonzero(output) == x.shape[0] * k


def test_abstopk_backward():
    """
    Test if gradients are only propagated through the active (kept) elements.
    """
    x = torch.tensor([1.0, 10.0, -8.0, 2.0], requires_grad=True)
    # Top 2 abs: 10.0, -8.0
    k = 2
    
    output = run_abstopk(x, k=k)
    loss = output.sum()
    loss.backward()
    
    # Gradients should be 1.0 for kept elements, 0.0 for pruned elements
    expected_grad = torch.tensor([0.0, 1.0, 1.0, 0.0])
    torch.testing.assert_close(x.grad, expected_grad)


def test_attention_with_sink(numpy_snapshot, q, k, v, mask, n_heads):    
    q, k, v = (rearrange(x, "b s (h d) -> b h s d", h=n_heads) for x in (q, k, v))
    
    mask = mask.unsqueeze(1).expand(-1, n_heads, -1, -1)

    d_head = q.shape[-1]
    sink_token = torch.ones((1, d_head)) * 10.0 
    
    output = run_attention_with_sink(Q=q, K=k, V=v, sink_token=sink_token, mask=mask)
    
    numpy_snapshot.assert_match(output, atol=1e-5)


def test_magnitude_pruning():
    """
    Test if the pruning function correctly zeroes out small weights across a simple model.
    """
    model = torch.nn.Sequential(
        torch.nn.Linear(10, 10, bias=False),
        torch.nn.Linear(10, 5, bias=False)
    )
    
    # Manually set weights to know distribution
    # Layer 1: 100 weights. Let's make 20 of them large (10.0) and 80 small (0.1)
    with torch.no_grad():
        model[0].weight.fill_(0.1)
        model[0].weight.view(-1)[:20].fill_(10.0)
        
        # Layer 2: 50 weights. All small (0.01)
        model[1].weight.fill_(0.01)
        
    total_params = 100 + 50
    
    # Prune 50% of weights. 
    # Total params = 150. Target non-zero = 75.
    # Our large weights are 20. The pruning should keep the 20 large ones 
    # and the largest 55 of the remaining ones (or random if tied).
    # To make it deterministic, let's set a gradient of values.
    for i, p in enumerate(model.parameters()):
        with torch.no_grad():
            # Fill with 0, 1, 2, ...
            p.view(-1).copy_(torch.arange(p.numel()).float() + (i * 1000))

    # Apply pruning: sparsity_level 0.4 means set smallest 40% to zero.
    sparsity_level = 0.4
    run_magnitude_pruning(model, sparsity_level=sparsity_level)
    
    # Check global sparsity
    total_zeros = 0
    total_elements = 0
    for p in model.parameters():
        total_zeros += torch.sum(p == 0).item()
        total_elements += p.numel()
        
    actual_sparsity = total_zeros / total_elements
    
    # Allow small floating point margin or discrete count margin
    assert abs(actual_sparsity - sparsity_level) < 0.05
    
    # Ensure large values are preserved (last elements in our arange init)
    assert model[1].weight.view(-1)[-1] != 0 # The largest value should be kept