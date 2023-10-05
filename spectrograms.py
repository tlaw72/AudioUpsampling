import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

def convert_wav_to_spectrogram(input_folder, output_folder):
    # Check if the provided folder paths exist
    if not os.path.exists(input_folder):
        print(f"The input folder '{input_folder}' does not exist.")
        return
    if not os.path.exists(output_folder):
        print(f"The output folder '{output_folder}' does not exist.")
        return
    
    count = 0

    # hq_path_train = "./audio_files/hq_train/hq"
    # lq_path_train = "./audio_files/hq_train/hq"
    # hq_path_val = "Spectrograms/val/hq_val"
    # lq_specs_train = os.listdir("Spectrograms/train/lq_train")
    # lq_specs_val = os.listdir("Spectrograms/val/lq_val")
    # print(f'Num files in lq_specs_train: {len(lq_specs_train)}')
    # print(f'Num files in lq_specs_val: {len(lq_specs_val)}')


    # Iterate through all files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".wav"):

            print(filename)
            file_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, f"{filename[:-4]}.png")

            # Load the audio file
            waveform, sample_rate = librosa.load(file_path, sr=None)

            # Convert to mono if needed
            if waveform.ndim > 1:
                waveform = np.mean(waveform, axis=1)

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
            plt.savefig(output_path, bbox_inches="tight", pad_inches=0)
            plt.close()

            count += 1
            # return
            if count %50 == 0:
              print(f'{count} spectrograms produced')

    print(f"{count} spectrograms have been generated for all .wav files.")
    # hq_specs_train = os.listdir(hq_path_train)
    # hq_specs_val = os.listdir(hq_path_val)
    # print(f'Num files in hq_specs_train: {len(hq_specs_train)}')
    # print(f'Num files in hq_specs_val: {len(hq_specs_val)}')



if __name__ == "__main__":
    # Usage example
    input_folder = "./audio_files/train/hq"  # Replace with the actual input folder path
    output_folder = "./spectrograms/train/hq"  # Replace with the actual output folder path
    convert_wav_to_spectrogram(input_folder, output_folder)