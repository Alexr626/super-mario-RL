import torch

def save_model(model, filename):
    """Save the model parameters to a file."""
    torch.save(model.state_dict(), filename)

def load_model(model, filename):
    """Load model parameters from a file."""
    model.load_state_dict(torch.load(filename))