def print_model_sanity_check(model):
    '''
    Prints a sanity check of the model's parameters, showing which layers are trainable and which are frozen.
    This is especially useful after applying fine-tuning modifications to ensure that the intended layers are frozen and the new head layers are trainable.
    '''
    print("\n" + "="*80)
    print(f"{'LAYER NAME':<50} | {'SHAPE':<15} | {'TRAINABLE'}")
    print("-"*80)
    
    trainable_params = 0
    frozen_params = 0
    
    for name, param in model.named_parameters():
        status = "✅ YES" if param.requires_grad else "❌ NO "
        print(f"{name:<50} | {str(list(param.shape)):<15} | {status}")
        
        if param.requires_grad:
            trainable_params += param.numel()
        else:
            frozen_params += param.numel()
            
    print("-"*80)
    print(f"Total Trainable Params: {trainable_params:,}")
    print(f"Total Frozen Params:    {frozen_params:,}")
    print("="*80 + "\n")