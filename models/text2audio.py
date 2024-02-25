import torch
import torch.nn as nn
import torch.optim as optim
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt

# Generator and Discriminator classes
class Generator(nn.Module):
    def __init__(self, latent_dim, text_embedding_dim, output_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim + text_embedding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
            nn.Tanh()
        )

    def forward(self, z, text_embedding):
        combined_input = torch.cat([z, text_embedding], dim=1)
        return self.model(combined_input)

class Discriminator(nn.Module):
    def __init__(self, input_dim, text_embedding_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim + text_embedding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x, text_embedding):
        combined_input = torch.cat([x, text_embedding], dim=1)
        return self.model(combined_input)

# Training parameters
latent_dim = 100
text_embedding_dim = 50
output_dim = 220500  # Assuming audio samples with a length of 1 second (44.1 kHz)
batch_size = 64
num_epochs = 1000

# Initialize models
generator = Generator(latent_dim, text_embedding_dim, output_dim)
discriminator = Discriminator(output_dim, text_embedding_dim)

# Loss functions and optimizers
criterion = nn.BCELoss()
optimizer_G = optim.Adam(generator.parameters(), lr=0.0002)
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002)

# Training loop (placeholder for a real dataset)
for epoch in range(num_epochs):
    # Placeholder: generate random noise as input to the generator and random text embeddings
    z = torch.randn(batch_size, latent_dim)
    text_embedding = torch.randn(batch_size, text_embedding_dim)

    # Generate synthetic audio samples
    generated_samples = generator(z, text_embedding)

    # Placeholder: use random noise as real audio samples and corresponding text embeddings
    real_samples = torch.randn(batch_size, output_dim)
    real_labels = torch.ones(batch_size, 1)
    fake_labels = torch.zeros(batch_size, 1)

    optimizer_D.zero_grad()

    real_outputs = discriminator(real_samples, text_embedding)
    fake_outputs = discriminator(generated_samples.detach(), text_embedding)

    loss_real = criterion(real_outputs, real_labels)
    loss_fake = criterion(fake_outputs, fake_labels)
    loss_D = loss_real + loss_fake

    loss_D.backward()
    optimizer_D.step()

    # Train the generator
    optimizer_G.zero_grad()
    fake_outputs = discriminator(generated_samples, text_embedding)
    loss_G = criterion(fake_outputs, real_labels)
    loss_G.backward()
    optimizer_G.step()

    # Print training progress
    if epoch % 100 == 0:
        print(f"Epoch [{epoch}/{num_epochs}], Loss D: {loss_D.item()}, Loss G: {loss_G.item()}")

# Generate audio from text
num_samples = 1
z = torch.randn(num_samples, latent_dim)
text_embedding = torch.randn(num_samples, text_embedding_dim)

with torch.no_grad():
    generated_samples = generator(z, text_embedding).numpy()

# Plot the generated audio waveform
plt.figure(figsize=(10, 4))
plt.plot(generated_samples[0])
plt.title('Generated Audio Waveform')
plt.xlabel('Sample Index')
plt.ylabel('Amplitude')
plt.show()

output_file = '/Users/harshdhiman/Documents/Research /Codes/AudioGAN/generated_audio/text2audio.wav'
sf.write(output_file, generated_samples[0], 44100, 'PCM_16')
