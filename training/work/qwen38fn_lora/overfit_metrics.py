"""Chunked, independently scored next-token metrics without full-context logits."""

import torch
import torch.nn.functional as F


@torch.no_grad()
def score_hidden(hidden, weight, labels, chunk=128):
    if hidden.ndim!=3 or hidden.shape[:2]!=labels.shape or hidden.shape[0]!=1:
        raise ValueError('expected one packed sequence with aligned hidden states and labels')
    hidden = hidden.reshape(-1, hidden.shape[-1])[:-1]
    target = labels.reshape(-1)[1:]
    valid = target != -100
    hidden, target = hidden[valid], target[valid]
    if not target.numel():
        raise ValueError('no supervised next tokens')
    loss = torch.zeros((), device=hidden.device, dtype=torch.float64)
    correct = torch.zeros((), device=hidden.device, dtype=torch.long)
    for offset in range(0, len(target), chunk):
        logits = F.linear(hidden[offset:offset+chunk], weight).float()
        wanted = target[offset:offset+chunk]
        loss += F.cross_entropy(logits, wanted, reduction='sum').double()
        correct += (logits.argmax(-1) == wanted).sum()
    return {'loss_sum': float(loss), 'correct': int(correct), 'tokens': len(target)}


def overfit_pass(record, exact_tokens=False):
    # Every example must pass; a long/easy sequence cannot hide a failed one.
    if len(record['examples'])!=4 or any(row['tokens']<=0 for row in record['examples']):
        raise ValueError('four nonempty examples required')
    return all(row['loss_sum']/row['tokens'] <= (0.0001 if exact_tokens else 0.02) and
               row['correct']/row['tokens'] >= (1.0 if exact_tokens else 0.995) for row in record['examples'])
