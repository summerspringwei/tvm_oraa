import time
import logging
import torch
import numpy as np
from tvm.meta_schedule.logging import get_logger
print(torch.__version__)
print(torch.__file__)
logger = get_logger("test_tensorboard")  # pylint: disable=invalid-name


preds = torch.tensor([0.8, -0.5, 1.5, 1.2], dtype=torch.float32)
label = torch.tensor([-0.5, 0.8, 1.2, 1.5], dtype=torch.float32)
# preds_indices = []
def pairwise_rank(preds: torch.Tensor, label: torch.Tensor):
    preds, label = preds[None, :], label[None, :]
    preds_sorted, preds_indices = torch.sort(preds, descending=True)
    print(preds_sorted, preds_indices)
    label_sorted, label_indices = torch.sort(label, descending=True)
    print(label_sorted, label_indices)
    true_sorted_by_preds = torch.gather(label, dim=1, index=preds_indices)
    # 1.2000,  1.5000, -0.5000,  0.8000
    print(true_sorted_by_preds)
    true_diffs = true_sorted_by_preds[:, :, None] - true_sorted_by_preds[:, None, :]
    print(true_diffs)
    # preds_indices, label_indices = preds_indices[None, :], label_indices[None, :]
    print(preds_indices, label_indices)
    preds_diff = preds_indices[:, :, None] - preds_indices[:, None, :]
    label_diff = label_indices[:, :, None] - label_indices[:, None, :]
    print(preds_diff)
    print(label_diff)
    num_diff = torch.relu(torch.negative(preds_diff * label_diff))
    pairwise = torch.count_nonzero(num_diff,)
    print(pairwise)

pairwise_rank(preds, label)

exit(0)

def lambda_rank_loss(  # pylint: disable=too-many-locals
    preds: "torch.Tensor",
    labels: "torch.Tensor",
    k: int = None,
    eps: float = 1e-10,
    sigma: float = 1.0,
) -> "torch.Tensor":
    """
    LambdaLoss: Metric-Driven Loss for Learning-to-Rank

    Parameters
    ----------
    preds : Tensor
        The predicted runtime for each candidate.
    labels : Tensor
        The measured runtime for each candidate.
    k : int
        Loss for top k.
        Default is None, which means computing all scores.
    eps : float
        The minimum value to the denominator and argument of log if they reach 0.
    sigma : float
        The scaling factor to the input of the sigmoid function.

    Returns
    -------
    loss : Tensor
        The lambda rank loss.
    """
    device = preds.device
    logger.info(preds.shape)
    logger.info(labels.shape)
    # Handle preds and label dim 0
    def dim0_to_dim1(t: torch.Tensor):
        if t.dim() > 0:
            return t
        else:
            return t.reshape([1])
    preds, labels = dim0_to_dim1(preds), dim0_to_dim1(labels)
    # [0.8, -0.5, 1.5, 1.2], [-0.5, 0.8, 1.2, 1.5]
    y_pred, y_true = preds[None, :], labels[None, :]
    # tensor([ 1.5000,  1.2000,  0.8000, -0.5000]) tensor([2, 3, 0, 1])
    y_pred_sorted, indices_pred = y_pred.sort(descending=True, dim=-1)
    # tensor([ 1.5000,  1.2000,  0.8000, -0.5000]) tensor([3, 2, 1, 0])
    y_true_sorted, _ = y_true.sort(descending=True, dim=-1)
    # [1.2, 1.5, -0.5, 0.8]
    true_sorted_by_preds = torch.gather(y_true, dim=1, index=indices_pred)
    
    true_diffs = true_sorted_by_preds[:, :, None] - true_sorted_by_preds[:, None, :]
    padded_pairs_mask = torch.isfinite(true_diffs) & (true_diffs > 0)
    ndcg_at_k_mask = torch.zeros(
        (y_pred.shape[1], y_pred.shape[1]), dtype=torch.bool, device=device
    )
    ndcg_at_k_mask[:k, :k] = 1
    true_sorted_by_preds.clamp_(min=0.0)
    y_true_sorted.clamp_(min=0.0)
    pos_idxs = torch.arange(1, y_pred.shape[1] + 1).to(device)
    D = torch.log2(1.0 + pos_idxs.float())[None, :]  # pylint: disable=invalid-name
    maxDCGs = torch.sum(  # pylint: disable=invalid-name
        ((torch.pow(2, y_true_sorted) - 1) / D)[:, :k], dim=-1
    ).clamp(min=eps)
    G = (torch.pow(2, true_sorted_by_preds) - 1) / maxDCGs[:, None]  # pylint: disable=invalid-name
    weights = torch.abs(
        torch.pow(D[:, :, None], -1.0) - torch.pow(D[:, None, :], -1.0)
    ) * torch.abs(G[:, :, None] - G[:, None, :])
    scores_diffs = (y_pred_sorted[:, :, None] - y_pred_sorted[:, None, :]).clamp(min=-1e8, max=1e8)
    scores_diffs[torch.isnan(scores_diffs)] = 0.0
    weighted_probs = (torch.sigmoid(sigma * scores_diffs).clamp(min=eps) ** weights).clamp(min=eps)
    losses = torch.log2(weighted_probs)
    masked_losses = losses[padded_pairs_mask & ndcg_at_k_mask]
    loss = -torch.sum(masked_losses)
    return loss

loss = lambda_rank_loss(
    torch.tensor([0.8, -0.5, 1.5, 1.2], dtype=torch.float32),
    torch.tensor([-0.5, 0.8, 1.2, 1.5], dtype=torch.float32))
print(loss)

exit(0)
a = torch.from_numpy(np.array([1,2,3,4]))
print(a.dim())
print(a[None, :])
b = torch.from_numpy(np.array([1]))
print(b.dim())
print(b[None, :])
c = torch.tensor(1)

print(c.dim())
# c = torch.reshape(c, [-1])
print(c.dim())
# print(c[None, :])
# print(len(c))
print(np.concatenate([a, c]))
exit(0)
import tensorflow as tf


def test_tensorboard():
    writer = tf.summary.create_file_writer("/tmp/board")
    for i in range(100):
        with writer.as_default():
            tf.summary.scalar('learning rate', i, step=i)
        time.sleep(1)
        writer.flush()


def test_logger():
    logger = logging.getLogger("TTTT")
    logger.warn("aaaaa")

