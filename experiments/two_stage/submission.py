"""Official prediction entry point.

`predict` compares the four supplied solutions and returns one index per
question. Learned weights, if any, come only from the checkpoint directory.
"""

from selector import choose, load_checkpoint, majority_choices


def predict(bank, arrays, checkpoint):
    """Return one Python integer in 0..3 per question, in bank order."""
    if checkpoint is None:
        return majority_choices(bank)
    model = load_checkpoint(checkpoint)
    return choose(bank, arrays['embeddings'], model)
