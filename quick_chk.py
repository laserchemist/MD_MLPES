from modules.data_formats import load_trajectory

traj = load_trajectory('augmented_training_data.npz')

if 'dipoles' in traj.metadata:
    dipoles = traj.metadata['dipoles']
    print(f"✅ Augmented data has {len(dipoles)} dipole moments")
    print(f"   Shape: {dipoles.shape}")
else:
    print("❌ No dipoles found")
