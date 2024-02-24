import torch
import torch.nn as nn
import torch.optim as optim
import librosa
import soundfile as sf
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# Define the Generator and Discriminator networks
class Generator(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
            nn.Tanh()
        )

    def forward(self, z):
        return self.model(z)

class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# Set the random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Hyperparameters
latent_dim = 100
output_dim = 220500  # Assuming audio samples with a length of 1 second (44.1 kHz)

# Initialize the Generator and Discriminator
generator = Generator(latent_dim, output_dim)
discriminator = Discriminator(output_dim)

# Loss functions and optimizers
criterion = nn.BCELoss()
optimizer_G = optim.Adam(generator.parameters(), lr=0.0002)
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002)

# Training loop
num_epochs = 1000
batch_size = 32

for epoch in range(num_epochs):
    # Generate random noise as input to the generator
    z = torch.randn(batch_size, latent_dim)

    # Generate synthetic audio samples
    generated_samples = generator(z)

    # Train the discriminator
    real_samples = torch.randn(batch_size, output_dim)
    real_labels = torch.ones(batch_size, 1)
    fake_labels = torch.zeros(batch_size, 1)

    optimizer_D.zero_grad()

    real_outputs = discriminator(real_samples)
    fake_outputs = discriminator(generated_samples.detach())

    loss_real = criterion(real_outputs, real_labels)
    loss_fake = criterion(fake_outputs, fake_labels)
    loss_D = loss_real + loss_fake

    loss_D.backward()
    optimizer_D.step()

    # Train the generator
    optimizer_G.zero_grad()
    fake_outputs = discriminator(generated_samples)
    loss_G = criterion(fake_outputs, real_labels)
    loss_G.backward()
    optimizer_G.step()

    # Print training progress
    if epoch % 100 == 0:
        print(f"Epoch [{epoch}/{num_epochs}], Loss D: {loss_D.item()}, Loss G: {loss_G.item()}")

torch.save(generator,'/Users/harshdhiman/Documents/Research /Codes/Audio/model/generator.pth')

def generate_audio(num_samples,latent_dim):
    z = torch.randn(num_samples, latent_dim)

    print("Generating samples")
    # Generate synthetic audio samples
    with torch.no_grad():
         generated_samples = generator(z).numpy()

    # Plot the generated audio waveform
    plt.figure(figsize=(10, 4))
    plt.plot(generated_samples[0])
    plt.title('Generated Audio Waveform')
    plt.xlabel('Sample Index')
    plt.ylabel('Amplitude')
    plt.show()

    output_file = 'generated_audio.wav'
    sf.write(output_file, generated_samples[0], 44100, 'PCM_16')
    return output_file

output_file=generate_audio(1,100)