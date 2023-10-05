#run this script
import os
from wav2spec import convert_wav_to_spectrogram, load_spectrograms
from generator import Generator, Discriminator
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import librosa
from genwav import generate_waveform
from train import train
from PIL import Image
import torchvision.transforms as transforms



def main():
    input_folder = "./spectrograms/train/lq_train"
    output_path = "./spectrograms/generated/train/spec_"

    # Call the convert_wav_to_spectrogram function
    lq_spectrograms = load_spectrograms(input_folder)
    # spectrograms = convert_wav_to_spectrogram("./audio_files/train/lq")
    print(lq_spectrograms[0][0].shape)

    # Initialize generator and discriminator
    generator = Generator(input_channels=4, output_channels=4, ngf=64, n_residual_layers=2, vision_transformer=True)
    discriminator = Discriminator(input_channels=4, ndf=64)

    # Train the models
    generator, discriminator = train(generator, discriminator, lq_spectrograms)

    # Test the models
    # test(generator, discriminator, dataloader)

    print("done training")
    # return

    # unit test with first element of downsampled audio spectrograms as input tensor
    for i, (input_tensor, input_sr, song_name) in enumerate(lq_spectrograms):
        img_output_path = output_path + song_name
        # input_tensor, input_sr = spectrograms[0]
        print("Input Spectrogram Shape:", input_tensor.shape)
        print("Input Spectrogram Sampling Rate:", input_sr)
        # print("Input Spectrogram :", input_tensor)

        # Pass the input tensor through the generator
        generated_spectrogram = generator(input_tensor.unsqueeze(0))
        print("Headed to discriminator")


        # Pass the generated spectrogram through the discriminator
        discriminator_output = discriminator(generated_spectrogram)

        print("Generated Spectrogram Shape:", generated_spectrogram.shape)
        print("Generated Spectrogram:", generated_spectrogram)
        print("Discriminator Output Shape:", discriminator_output.shape)

        # plt.figure(figsize=(10,4))
        # generated_spectrogram = generated_spectrogram.squeeze().permute(1,2,0)
        # # plt.plot(generated_spectrogram)
        
        # plt.imshow(generator_output)

        # Assuming generator_output is your generated spectrogram tensor with shape [1, 256, 256, 864]
        generator_output = generated_spectrogram.squeeze()  # Remove singleton dimensions
        generator_output = generator_output.permute(1, 2, 0)  # Reshape to [256, 256, 864]
        # generator_output = generator_output.reshape(4, -1)  # Reshape to [256, -1]

        # Convert the tensor to a NumPy array
        generator_output = generator_output.detach().numpy()   
        generator_output = (generator_output + 1) / 2 # Scale between 0 and 1

        # Plot the mel-spectrogram
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(generator_output, sr=input_sr, hop_length=512, x_axis='time', y_axis='mel')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Generated Mel-spectrogram')
        print("Trying to save plot")
        plt.savefig(img_output_path, bbox_inches="tight", pad_inches=0)
        print("Plot saved")
        plt.close()

        out_spectrograms = []
        out_spectrograms.append(generated_spectrogram)

        sr = input_sr  # Sample rate
        output_folder = "./outwaveforms"  # Output folder

        print("I'm here and completed the plot")
        if i == 50: break

        # generate_waveform(out_spectrograms, sr, output_folder)

if __name__ == "__main__":
    main()
