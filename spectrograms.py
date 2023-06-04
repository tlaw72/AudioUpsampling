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
    # hq_path = "/spectrograms/hq"
    hq_path_train = "Spectrograms/hq/train"
    hq_path_val = "Spectrograms/hq/val"
    # hq_specs = os.listdir(hq_path)
    lq_specs_train = os.listdir("Spectrograms/lq/train")
    lq_specs_val = os.listdir("Spectrograms/lq/val")
    print(f'Num files in lq_specs_train: {len(lq_specs_train)}')
    print(f'Num files in lq_specs_val: {len(lq_specs_val)}')


    # Iterate through all files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".wav"):
            print(filename)
            if str(filename[:-4]) + ".png" in lq_specs_train:
                output_folder = hq_path_train
            elif str(filename[:-4]) + ".png" in lq_specs_val:
                output_folder = hq_path_val
            else: 
                print("File not found in any lq folders")
                break
            if str(filename[:-4]) + ".png" in os.listdir(output_folder):
                # print("found repeat")
                continue
            file_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, f"{filename[:-4]}.png")

            # Load the audio file and compute the spectrogram
            y, sr = librosa.load(file_path)
            spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=1024, n_fft=2048)
            spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)

            # Plot and save the spectrogram
            plt.figure(figsize=(10, 4), dpi=600)
            librosa.display.specshow(spectrogram_db, sr=sr, x_axis="time", y_axis="mel")
            plt.axis("off")  # Turn off axis labels and ticks
            plt.tight_layout()
            plt.savefig(output_path, bbox_inches="tight", pad_inches=0)
            plt.close()
            count += 1
            if count %50 == 0:
              print(f'{count} spectrograms produced')

    print(f"{count} spectrograms have been generated for all .wav files.")



if __name__ == "__main__":
    # Usage example
    input_folder = "../hq_clips"  # Replace with the actual input folder path
    output_folder = "Spectrograms/hq/train"  # Replace with the actual output folder path
    convert_wav_to_spectrogram(input_folder, output_folder)