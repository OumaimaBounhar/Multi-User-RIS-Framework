import scipy.stats
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

#---------------------------------------- Experiment helpers ------------------------------------------------------------------------------
def fit_noise(filename: str, feedback, channel, parameters, max_samples=1000):
    print("Fitting real noise to Gaussian noise")

    size_codebooks, _ = parameters.get_codebook_parameters()
    noise_results = []

    for snr in parameters.snr_values:
        print(f"\nProcessing SNR = {snr} dB...")
        
        # Set the SNR for the current iteration
        parameters.set_SNR(snr)
        parameters.std_noise = np.sqrt(10 ** (-snr / 10))
        parameters.set_noise(mean_noise=0, std_noise=parameters.std_noise)

        data = []
        noise_contribution = []

        for sample in tqdm(range(max_samples), desc=f"SNR {snr} dB"):
            channel.new_channel()
            for n_communications in range(size_codebooks[1]):
                feedback.transmit(n_communications, codebook_used=1)
                RSE = feedback.get_feedback(noise=True)
                RSE_no_noise = feedback.get_feedback(noise=False)

                noise_value = RSE - RSE_no_noise
                data.append(noise_value)
                noise_contribution.append(np.log(RSE_no_noise / noise_value if noise_value != 0 else 0))

        # Fit noise to a Gaussian distribution
        mu, sigma = scipy.stats.norm.fit(data)
        noise_results.append([snr, mu, sigma])

        # Save parameters for this SNR
        res_df = pd.DataFrame([[snr, mu, sigma]], columns=["SNR (dB)", "Mean", "Std"])
        res_df.to_csv(f"{filename}/Noise_parameters_snr_{snr}.csv", index=False)

        # Plot histogram of noise contribution
        plt.figure(figsize=(10, 6))
        plt.hist(noise_contribution, bins=50, density=True, alpha=0.6, color='b', edgecolor='black', label='Noise Contribution')
        plt.title(f'Histogram of Noise Contribution (SNR={snr} dB)')
        plt.xlabel('Noise Contribution')
        plt.ylabel('Density')
        plt.legend()
        plt.grid(True)

        # Save histogram for this SNR
        plt.savefig(f"{filename}/Noise_histogram_snr_{snr}.png")
        plt.close()

    # Save all noise fitting results in one file
    noise_results_df = pd.DataFrame(noise_results, columns=["SNR (dB)", "Mean", "Std"])
    noise_results_df.to_csv(f"{filename}/Noise_parameters_all.csv", index=False)
    # Canonical file expected by Store/noise loading.
    noise_results_df.loc[:, ["Mean", "Std"]].iloc[[0]].to_csv(
        f"{filename}/Noise_parameters.csv", index=False
    )

    first_row = noise_results_df.iloc[0]
    return float(first_row["Mean"]), float(first_row["Std"])

#------------------------------------------------------------------------------------------------------------------------------------------------
