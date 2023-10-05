import torch
import librosa
import numpy as np
import os
import io
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms

def convert_wav_to_spectrogram(input_folder):
    spectrograms = []
    for filename in os.listdir(input_folder):
        if filename.endswith(".wav"):
            print(filename)
            file_path = os.path.join(input_folder, filename)

            # Load the audio file
            waveform, sample_rate = librosa.load(file_path, sr=None)

            # Convert to mono if needed
            if waveform.ndim > 1:
                waveform = np.mean(waveform, axis=1)

            spectrogram = librosa.feature.melspectrogram(y=waveform, sr=sample_rate, n_mels=1024, n_fft=2048)
            # print(spectrogram.shape)

            # Apply mel-spectrogram transformation
            mel_spectrogram = librosa.feature.melspectrogram(
                y=waveform,
                sr=sample_rate,
                n_fft=2048,
                hop_length=512,
                n_mels=256,
                fmin=0.0,
                fmax=None
            )
            # print(mel_spectrogram.shape)

            # Convert to decibels (log scale)
            mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
            # print(mel_spectrogram_db.shape)

            # Normalize the mel-spectrogram to the range [-1, 1]
            normalized_mel_spectrogram = (mel_spectrogram_db - np.min(mel_spectrogram_db))
            denominator = np.max(normalized_mel_spectrogram)
            if denominator != 0:
                normalized_mel_spectrogram /= denominator
            normalized_mel_spectrogram = normalized_mel_spectrogram * 2 - 1

            plt.figure(figsize=(10, 4), dpi=600)
            librosa.display.specshow(normalized_mel_spectrogram, sr=sample_rate, x_axis="time", y_axis="mel")
            plt.axis("off")  # Turn off axis labels and ticks
            plt.tight_layout()
            img_buf = io.BytesIO()
            plt.savefig(img_buf, format='png')

            im = Image.open(img_buf)

            transform = transforms.Compose([
                transforms.ToTensor()
            ])
            
            img_tensor = transform(im)
            # tensor_mel_spectrogram = torch.from_numpy(np.array(img_tensor))
            # im.show(title="My Image")
            # print(img_tensor)
            print(img_tensor.min())
            print(img_tensor.max())
            img_buf.close()
            plt.close()

            # Convert to tensor
            # tensor_mel_spectrogram = torch.from_numpy(normalized_mel_spectrogram).unsqueeze(0).unsqueeze(0).float()
            print(img_tensor.shape)

            # Append the tensor to the list
            spectrograms.append((img_tensor, sample_rate))

    return spectrograms

def load_spectrograms(input_folder):
    loaded_specs = []
    specs = os.listdir(input_folder)
    count = 1
    for spec in specs:
        img = Image.open(os.path.join(input_folder, spec))
        transform = transforms.Compose([
                transforms.ToTensor()
            ])
            
        img_tensor = transform(img)
        sr = 44100
        if input_folder.endswith("lq"):
            sr = 8000
        loaded_specs.append((img_tensor, sr, spec))
        if count % 6 == 0: print(f'{count} images loaded')
        if count % 6 == 0: break
        count += 1
    return loaded_specs