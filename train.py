import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import librosa
from torch.utils.data import Dataset, DataLoader
from wav2spec import convert_wav_to_spectrogram, load_spectrograms
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr


class SpectrogramDataset(Dataset):
    def __init__(self, lq_spectrograms, hq_spectrograms):
        self.lq_spectrograms = lq_spectrograms
        self.hq_spectrograms = hq_spectrograms

    def __getitem__(self, index):
        spectrogram = (self.lq_spectrograms[index], self.hq_spectrograms[index])
        return spectrogram

    def __len__(self):
        return len(self.lq_spectrograms)


#Train Generator and Discriminator
def train(generator, discriminator, lq_spectrograms):

    # Define the loss function (e.g., Binary Cross Entropy)
    criterion = nn.BCELoss()

    # Define the optimizers for generator and discriminator
    generator_optimizer = torch.optim.Adam(generator.parameters(), lr=0.001)
    discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.001)

    # Training loop
    num_epochs = 2
    #batch_size = 1
    discriminator_losses = []
    generator_losses = []
    accuracies = []
    
    dataloader = createDataloader(lq_spectrograms)

    for epoch in range(num_epochs):
        g_loss = []
        d_loss = []
        accuracy = []
        count = 1
        for real_data, downsampled_data in dataloader:  # Iterate over real high-quality mel spectrograms from the dataset
            # print(f"real_data: {real_data}, downsampled_data: {downsampled_data}")
            print(f"real_data type: {type(real_data)}, downsampled_data: {type(downsampled_data)}")
            real_data = real_data.to(torch.float32)
            # print("real_data", real_data.shape)
            # Train the discriminator
            discriminator.zero_grad()

            # Generate fake mel spectrogram from the generator
            fake_data = generator(downsampled_data)  # Generate a high-quality mel spectrogram from the downsampled mel spectrogram

            # fake_data = fake_data.squeeze()  # Remove singleton dimensions
            # print(f'after singleton: {fake_data.shape}')
            # fake_data = fake_data.permute(1, 0, 2)  # Reshape to [256, 256, 864]
            # print(f'after permute: {fake_data.shape}')
            # fake_data = fake_data.reshape(256, -1)  # Reshape to [256, -1]
            # print(f'after reshape: {fake_data.shape}')
            # plot_spec(fake_data)
            # plot_spec(real_data.squeeze())
            # print(real_data.shape)
            # print(fake_data.detach().shape)
            # print(real_data.view(1,256,256,160).shape)

            # Calculate the discriminator's predictions for real and fake data
            real_pred = discriminator(real_data)
            fake_pred = discriminator(fake_data.detach())

            # Calculate the discriminator's loss
            real_labels = torch.ones_like(real_pred)
            fake_labels = torch.zeros_like(fake_pred)
            discriminator_loss = criterion(real_pred, real_labels) + criterion(fake_pred, fake_labels)
            d_loss.append(discriminator_loss.item())
            

            # Update the discriminator
            discriminator_loss.backward()
            discriminator_optimizer.step()

            # Train the generator
            generator.zero_grad()

            # Generate fake mel spectrogram from the generator
            fake_data = generator(real_data)  # Generate a high-quality mel spectrogram from the downsampled mel spectrogram

            # Calculate the discriminator's predictions for the generated data
            fake_pred = discriminator(fake_data)

            # Calculate the generator's loss
            generator_loss = criterion(fake_pred, real_labels)
            g_loss.append(generator_loss.item())
            

            acc = psnr(fake_data.detach().numpy(), real_data.detach().numpy())
            accuracy.append(acc)

            print(f"generator loss: {generator_loss}, discriminator loss: {discriminator_loss}, accuracy: {acc}")

            # Update the generator
            generator_loss.backward()
            generator_optimizer.step()
            if count % 50 == 0:
                generator_losses.append(np.mean(g_loss))
                discriminator_losses.append(np.mean(d_loss))
                accuracies.append(np.mean(accuracy))
                print(f"Epoch [{epoch + 1}/{num_epochs}], Generator Loss: {np.mean(g_loss)}, Discriminator Loss: {np.mean(d_loss)}, Accuracy: {np.mean(accuracy)}")
                g_loss = []
                d_loss = []
                accuracy = []
            count += 1

        # Print the losses for monitoring the training progress
        # print(f"Epoch [{epoch + 1}/{num_epochs}], Generator Loss: {np.mean(g_loss)}, Discriminator Loss: {np.mean(discriminator_loss)}, Accuracy: {np.mean(accuracy)}")
    
    
    plot_acc(accuracies)
    plot_loss(generator_losses, discriminator_losses)
    return (generator, discriminator)

def plot_spec(data):

    # Convert the tensor to a NumPy array
    generator_output = data.detach().numpy()

    # Plot the mel-spectrogram
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(generator_output, sr=44100, hop_length=512, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Generated Mel-spectrogram')
    plt.show()

def plot_loss(gen_loss, disc_loss):

    output_path = "./plots/loss_graph.png"

    plt.plot(gen_loss, label='Generator Loss')
    plt.plot(disc_loss, label='Discriminator Loss')
    plt.xlabel('Time (Per 100 images)')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(output_path)
    plt.close()

def plot_acc(acc):

    output_path = "./plots/accuracy_graph.png"

    plt.plot(acc, label='Accuracy')
    plt.xlabel('Time (Per 100 images)')
    plt.ylabel('Model Accuracy')
    plt.legend()
    plt.savefig(output_path)
    plt.close()
    

def createDataloader(lq_spectrograms, train=True, specs=True):

    hq_mel_spectrograms =  None
    if specs:
        hq_audio_files = "./spectrograms/train/hq_train" if train else "./spectrograms/val/hq_val"
        lq_audio_files = "./spectrograms/train/lq_train" if train else "./spectrograms/val/lq_val"
        hq_mel_spectrograms = load_spectrograms(hq_audio_files)
        # lq_mel_spectrograms = load_spectrograms(lq_audio_files)
    else:
        hq_audio_files = "./audio_files/train/hq" if train else "./audio_files/val/hq"
        lq_audio_files = "./audio_files/train/lq" if train else "./audio_files/val/lq"
        hq_mel_spectrograms = convert_wav_to_spectrogram(hq_audio_files)
        # lq_mel_spectrograms = convert_wav_to_spectrogram(lq_audio_files)

    hq_mel_spectrograms = list(zip(*hq_mel_spectrograms))[0]
    lq_mel_spectrograms = list(zip(*lq_spectrograms))[0]  

    dataset = SpectrogramDataset(lq_mel_spectrograms, hq_mel_spectrograms)
    #batch_size should be 32
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    return dataloader