import argparse
import time
from typing import Any

import torch
from jaxtyping import Float, Int
from torch import nn, Tensor

from cs336_basics.optimizer import AdamW
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def generate_data(batch_size: int, sequence_length: int, vocab_size: int) -> tuple[Int[Tensor, " ... sequence_length"], Float[Tensor, " ... sequence_length vocab_size"]]:
    """Generates a random batch of input and output data
    """
    data: Tensor = torch.randint(low=0, high=vocab_size, size=(batch_size, sequence_length+1))

    return data[:, :sequence_length].to(device), data[:, 1:sequence_length+1].to(device)

def sync_gpu() -> None:
    """Synchronizes CUDA device if available to ensure accurate timing."""
    if device.type == "cuda":
        torch.cuda.synchronize()


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description="Parses model parameters.")
    parser.add_argument("--vocab_size", type=int, default=10_000, help="The number of unique items in the output vocabulary to be predicted.")
    parser.add_argument("--context_length", type=int, default=64, help="The maximum number of tokens to process at once.")
    parser.add_argument("--d_model", type=int, default=768, help="The dimensionality of the model embeddings and sublayer outputs.")
    parser.add_argument("--num_layers", type=int, default=12, help="The number of Transformer layers to use.")
    parser.add_argument("--num_heads", type=int, default=12, help="Number of heads to use in multi-headed attention. `d_model` must be evenly divisible by `num_heads`.")
    parser.add_argument("--d_ff", type=int, default=3072, help="Dimensionality of the feed-forward inner layer (section 3.3).")
    arg_dict: dict[str, Any] = vars(parser.parse_args())

    # Training config
    warmup_steps: int = 5
    training_steps: int = 10
    batch_size = 64

    print(f"model arguments: {arg_dict}")
    print(f"training arguments: warmup_steps={warmup_steps}, training_steps={training_steps}, batch_size={batch_size}")

    # Initialize model and optimizer
    model: nn.Module = BasicsTransformerLM(**arg_dict).to(device)
    optimizer: torch.optim.Optimizer = AdamW(params=model.parameters())

    # Execute steps
    print("warming up ...")
    for warmup_step in range(warmup_steps):
        model.train()
        x, y = generate_data(batch_size, arg_dict["context_length"], arg_dict["vocab_size"])
        logits = model(x)
        loss = cross_entropy(logits, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)


    # Main training steps, gather stats
    forward_times = []
    loss_times = []
    backward_times = []
    optimizer_times = []

    model.train()
    for step in range(training_steps):
        optimizer.zero_grad()
        input_ids, targets = generate_data(batch_size, arg_dict["context_length"], arg_dict["vocab_size"])
        
        # 1. Forward Pass
        sync_gpu()
        start = time.perf_counter()
        logits = model(input_ids)
        sync_gpu()
        forward_times.append(time.perf_counter() - start)
        
        # 2. Loss Computation
        sync_gpu()
        start = time.perf_counter()
        loss = cross_entropy(logits, targets)
        sync_gpu()
        loss_times.append(time.perf_counter() - start)
        
        # 3. Backward Pass
        sync_gpu()
        start = time.perf_counter()
        loss.backward()
        sync_gpu()
        backward_times.append(time.perf_counter() - start)

        # 4. Optimizer Step
        sync_gpu()
        start = time.perf_counter()
        optimizer.step()
        sync_gpu()
        optimizer_times.append(time.perf_counter() - start)

        print(f"Step {step+1}/{training_steps} completed. Loss: {loss.item():.4f}")

    # --- Compute and Print Final Performance Metrics ---
    print(f"\n=== Performance Metrics Across {training_steps} Steps ===")
    components = {
        "Forward Pass": torch.tensor(forward_times),
        "Loss Computation": torch.tensor(loss_times),
        "Backward Pass": torch.tensor(backward_times),
        "Optimizer Step": torch.tensor(optimizer_times)
    }

    for name, timings in components.items():
        mean_time = timings.mean().item()
        std_time = timings.std().item() if len(timings) > 1 else 0.0
        print(f"{name:20} -> Mean: {mean_time:.6f}s | Std: {std_time:.6f}s")
