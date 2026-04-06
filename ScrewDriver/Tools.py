import torch
class AdaptiveSpikeScheduler:
    def __init__(self, optimizer, base_lr, min_lr=1e-6, spike_tolerance=1.5, plateau_patience=5, drop_factor=0.5):
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.min_lr = min_lr  # NEW: The hard floor to prevent paralysis
        self.best_loss = float('inf')
        self.patience_counter = 0
        self.spike_tolerance = spike_tolerance 
        self.plateau_patience = plateau_patience
        self.drop_factor = drop_factor

    def step(self, current_loss):
        if current_loss > self.best_loss * self.spike_tolerance:
            self._multiply_lr(self.drop_factor)
            print(f"      [!] Manifold Spike! Slashed LR to {self._get_lr():.2e}")
            self.best_loss = current_loss 
            self.patience_counter = 0
            return

        if current_loss < self.best_loss * 0.98:
            self.best_loss = current_loss
            self.patience_counter = 0
        else:
            self.patience_counter += 1
            
        if self.patience_counter >= self.plateau_patience:
            self._multiply_lr(self.drop_factor)
            print(f"      [-] Loss Plateau. Fine-tuning LR to {self._get_lr():.2e}")
            self.patience_counter = 0
            
    def _multiply_lr(self, factor):
        for param_group in self.optimizer.param_groups:
            # Drop the LR, but NEVER let it fall below min_lr
            new_lr = max(param_group['lr'] * factor, self.min_lr)
            param_group['lr'] = new_lr
            
    def _get_lr(self):
        return self.optimizer.param_groups[0]['lr']

    def reset_phase(self):
        """NEW: Wipes the memory and restores the base learning rate for a new phase."""
        self.best_loss = float('inf')
        self.patience_counter = 0
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.base_lr
            
def inject_weights(model, layer_idx: int, delta_W: torch.Tensor):
    model.encoder.layer[layer_idx].attention.output.dense.weight.data.add_(delta_W)

def remove_weights(model, layer_idx: int, delta_W: torch.Tensor):
    model.encoder.layer[layer_idx].attention.output.dense.weight.data.sub_(delta_W)