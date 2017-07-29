settings = {
    "learning_rate": 1e-4,
    "memory_size": 1000000,                 # 1 Million frames in memory
    "epsilon_start": 1.0,                         # Start of epsilon decent
    "epsilon_end": 0.0,                     # End of epsilon decent
    "epsilon_steps": 1,                     # Epsilon steps
    "exploration_wins": 0,                  # Number of victories using random moves before starting epsilon phase
    "use_training_data": False,
    "batch_size": 8,
    "discount_factor":  0.99


}