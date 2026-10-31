"""Public representation validation entry point."""
def evaluate(checkpoint,output,device='cpu',split='validation'):
    from .validation import representation
    return representation(checkpoint,output,device,split,subset=12)
