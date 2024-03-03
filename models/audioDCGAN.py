import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import torchaudio
from torchaudio.transforms import Spectrogram, AmplitudeToDB
import torchvision.utils as vutils

# Set random seed for reproducibility
torch.manual_seed(42)

# Load audio data (replace 'your_audio_file.wav' with your actual file)
audio_file = r'/Users/harshdhiman/Documents/Research /Codes/AudioGAN/sample_audio/hours.wav'
waveform, sr = torchaudio.load(audio_file)

# Spectrogram parameters
n_fft = 400
hop_length = 160
spec_transform = Spectrogram(n_fft=n_fft, hop_length=hop_length)
db_transform = AmplitudeToDB()

# Function to generate spectrograms from audio signals
def generate_spectrogram(audio_signal):
    spectrogram = spec_transform(audio_signal)
    spectrogram = db_transform(spectrogram)
    return spectrogram

# Function to generate spectrograms from audio signals
# Generate spectrograms from the audio data
segment_length = hop_length * 50
num_segments = waveform.size(1) // segment_length
spectrograms = [generate_spectrogram(waveform[:, i*segment_length:(i+1)*segment_length]) for i in range(num_segments)]

# Normalize spectrograms between -1 and 1
spectrograms = [(spec - spec.min()) / (spec.max() - spec.min()) * 2 - 1 for spec in spectrograms]

# Convert the list of spectrograms to a PyTorch tensor
spectrograms = torch.stack(spectrograms).unsqueeze(1)  # Add a channel dimension

# Reshape the tensor to remove unnecessary dimensions
spectrograms = spectrograms.view(-1, 1, 201, 51)



# Generator model
class Generator(nn.Module):
    def __init__(self, latent_dim, channels):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.channels = channels

        self.main = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)

class Discriminator(nn.Module):
    def __init__(self, channels):
        super(Discriminator, self).__init__()
        self.channels = channels

        self.main = nn.Sequential(
            nn.Conv2d(channels, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input.squeeze(2))  # Squeeze the extra dimension

# Choose device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize models
latent_dim = 100
channels = 1  # Number of channels in the spectrogram
generator = Generator(latent_dim, channels)
discriminator = Discriminator(channels)

# Loss function and optimizers
criterion = nn.BCEWithLogitsLoss()
optimizer_g = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_d = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

# ...

# Training the GAN
num_epochs = 1000
batch_size = 16

real_label = torch.ones((batch_size, 1, 1, 1), device=device)
fake_label = torch.zeros((batch_size, 1, 1, 1), device=device)

fixed_noise = torch.randn(64, latent_dim, 1, 1, device=device)

for epoch in range(num_epochs):
    for i in range(0, spectrograms.size(0), batch_size):
        # Discriminator training with real data
        discriminator.zero_grad()
        real_data = spectrograms[i:i + batch_size].to(torch.float32).to(device)
        label = real_label[:real_data.size(0)].expand_as(discriminator(real_data))
        output = discriminator(real_data)
        loss_real = criterion(output, label)
        loss_real.backward()

        # Discriminator training with fake data
        noise = torch.randn(batch_size, latent_dim, 1, 1, device=device)
        fake_data = generator(noise)
        label = fake_label[:fake_data.size(0)].expand_as(discriminator(fake_data.detach()))
        output = discriminator(fake_data.detach())
        loss_fake = criterion(output, label)
        loss_fake.backward()

        optimizer_d.step()

        # Generator training
        generator.zero_grad()
        label = real_label[:fake_data.size(0)].expand_as(discriminator(fake_data))
        output = discriminator(fake_data)
        loss_gen = criterion(output, label)
        loss_gen.backward()

        optimizer_g.step()

    
    if epoch % 100 == 0:
            print(f"Epoch {epoch}/{num_epochs}, "
                  f"Loss D: {loss_real.item() + loss_fake.item()}, Loss G: {loss_gen.item()}")

    # Save generated images at the end of each epoch
    #with torch.no_grad():
    #    fake_samples = generator(fixed_noise).detach().cpu()
    #    fake_grid = vutils.make_grid(fake_samples, normalize=True, nrow=8)
    #    plt.imshow(fake_grid.permute(1, 2, 0))
    #    plt.axis('off')
    #    plt.show()


# Generate and save example spectrogram
with torch.no_grad():
    noise = torch.randn(1, latent_dim, 1, 1, device=device)
    generated_spectrogram = generator(noise).squeeze().cpu().numpy()
    generated_spectrogram = (generated_spectrogram + 1) / 2  # Denormalize to [0, 1]

    # Display the generated spectrogram
    plt.figure(figsize=(10, 4))
    plt.imshow(generated_spectrogram, cmap='inferno', origin='lower')
    plt.title('Generated Spectrogram')
    plt.colorbar(format='%+2.0f dB')
    plt.show()

    # Convert the generated spectrogram to audio and save
    generated_waveform = torchaudio.transforms.InverseSpectrogram()(generated_spectrogram)
    torchaudio.save('generated_audio.wav', generated_waveform, sr)
