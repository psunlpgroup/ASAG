import torch

def sort_batch(batch, targets, lengths):
    # print("batch: ", batch)
    # print("targets: ", targets)
    # print("lengths: ", lengths)

    seq_lengths, perm_idx = lengths.sort(0, descending=True)
    # print("seq_lengths: ", seq_lengths)
    # print("perm_idx: ", perm_idx)

    seq_tensor = batch[perm_idx.squeeze()]
    # print("seq_tensor: ", seq_tensor)
    # print("seq_tensor.size: ", seq_tensor.size())

    target_tensor = targets[perm_idx.squeeze()]
    # print("target_tensor.size: ", target_tensor.size())

    return seq_tensor, target_tensor, seq_lengths
