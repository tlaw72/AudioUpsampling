import librosa
import numpy as np
import torch
import os
import soundfile as sf

def generate_waveform(out_spectrograms, sr, output_folder):
    for idx, spectrogram in enumerate(out_spectrograms):
        print("I'm here I just entered generate waveform")

        # Assuming "spectrogram" contains the generated spectrogram tensor

        # Reshape the spectrogram tensor
        spectrogram = spectrogram.squeeze(0).squeeze(0)

        # Convert spectrogram back to power scale
        mel_spectrogram_db = librosa.db_to_power(spectrogram.detach().numpy())

        print("Before inversion: ")

        # Invert the mel-spectrogram to obtain the linear spectrogram
        mel_spectrogram = librosa.feature.inverse.mel_to_stft(mel_spectrogram_db)

        print("Before conversion: ")

        # Convert the linear spectrogram back to the audio waveform
        waveform = librosa.griffinlim(mel_spectrogram)

        # Convert waveform to Torch tensor
        #waveform_tensor = torch.from_numpy(waveform).unsqueeze(0)

        # Create the output directory if it doesn't exist
        output_dir = output_folder
        os.makedirs(output_dir, exist_ok=True)

        # Specify the output file name
        output_file = "output" + str(idx) + ".wav"

        # Build the output path
        output_path = os.path.join(output_dir, output_file)
        print("output_path", output_path)

        print("I'm here right before saving the audio")
        print(waveform.shape)

        # Save the waveform as a WAV file
        sf.write(output_path, waveform.T, sr, format='WAV')