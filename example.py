from modules.ml_pes import MLPESConfig, MLPESTrainer, train_pes
from modules.data_formats import load_trajectory

# Load training data
trajectory = load_trajectory('water_trajectory.npz')

# Configure ML-PES
config = MLPESConfig(
    model_type='kernel_ridge',       # or 'neural_network'
    descriptor_type='coulomb_matrix',
    train_forces=True,
    kernel='rbf',
    kernel_params={'gamma': 0.1}
)

# Train model
trainer = MLPESTrainer(config)
trainer.train(trajectory)

# Save model
trainer.save('water_pes_model.pkl')

# Predict energy for new geometry
energy = trainer.predict(symbols, coordinates)