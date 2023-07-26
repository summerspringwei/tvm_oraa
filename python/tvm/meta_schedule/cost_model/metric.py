# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""Cost model metrics for meta schedule"""
import numpy as np  # type: ignore
import torch

def max_curve(trial_scores: np.ndarray) -> np.ndarray:
    """f(n) = max([s[i] fo i < n])

    Parameters
    ----------
    trial_scores : List[float]
        the score of i-th trial

    Returns
    -------
    curve : np.ndarray
        A vector, the max-curve function values
    """
    ret = np.empty(len(trial_scores))
    keep = -1e9
    for i, score in enumerate(trial_scores):
        keep = max(keep, score)
        ret[i] = keep
    return ret


def pairwise_rank_error_count(preds: torch.Tensor, label: torch.Tensor):
    """Compute number of pairwise mis-ranked

    Equal to the following code:
    count = 0
    for i in len(preds):
        for j len(preds):
            if ((preds_indices[i]>preds_indices[j]) ^ (label_indices[i]>label_indices[j]):
                count += 1 
    count /= 2
    Parameters
    ----------
    preds: torch.Tensor
        The prediction scores of the cost model
    label: torch.Tensor
        The true scores
    
    Returns
    -------
    count: The number of mis-ranked pairs
    """
    preds, label = preds[None, :], label[None, :]
    _, preds_indices = torch.sort(preds, descending=True)
    _, label_indices = torch.sort(label, descending=True)
    preds_compare = preds_indices[:, :, None] - preds_indices[:, None, :]
    label_compare = label_indices[:, :, None] - label_indices[:, None, :]
    # We use relu(multiplication(a, b)) to replace xor operator
    num_diff = torch.relu(torch.negative(preds_compare * label_compare))
    count = torch.count_nonzero(num_diff)
    count = count / 2
    return count


def top_k_intersection_count(preds: torch.Tensor, label: torch.Tensor, k: int = 10):
    """ Compute how many elements are intersection between preds and labels top_k elements

    Parameters
    ----------
    preds: torch.Tensor
        The prediction scores of the cost model
    label: torch.Tensor
        The true scores
    k: Int
        The top k elements to choose
    
    Returns
    -------
    count: The number of intersection elements
    """
    _, preds_indices = torch.sort(preds, descending=True)
    _, label_indices = torch.sort(label, descending=True)
    top_k_preds_indices, top_k_label_indices = preds_indices[:k], label_indices[:k]
    set_preds_indices, set_label_indices = set(top_k_preds_indices.tolist()), set(top_k_label_indices.tolist())
    out = set_preds_indices.intersection(set_label_indices)
    return len(out)


def top_one_performance_gap(preds: torch.Tensor, label: torch.Tensor):
    """Evaluate the performance gap between top-1 of preds and top-1 of label

    Parameters
    ----------
    preds: torch.Tensor
        The prediction scores of the cost model
    label: torch.Tensor
        The true scores
    
    Returns
    -------
    degration: Performance degration
    """
    _, preds_indices = torch.sort(preds, descending=True)
    _, label_indices = torch.sort(label, descending=True)
    top_1_preds_indices, top_1_label_indices = preds_indices[0], label_indices[0]
    degration = (label[top_1_label_indices] - label[top_1_preds_indices]) / label[top_1_label_indices]
    return degration