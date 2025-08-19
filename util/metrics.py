def print_model_param_counts(model, name="Model"):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    print(f"{name} has {sum(p.numel() for p in model.parameters() if p.requires_grad)} parameters!")
    print(f"Trainable:   {trainable_params:,}")
    print(f"Frozen:      {frozen_params:,}")
    print("-" * 40)
