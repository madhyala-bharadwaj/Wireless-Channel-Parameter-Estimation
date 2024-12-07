import numpy as np
import pandas as pd

def generate_channel_data(num_samples=1000, time_steps=5):
    np.random.seed(42)
    data = []

    for _ in range(num_samples):
        # Base parameters with periodic environmental effects
        signal_strength = np.random.uniform(0.5, 2.5, size=time_steps) + 0.2 * np.sin(np.linspace(0, np.pi, time_steps))
        noise_level = np.random.uniform(0.1, 0.6, size=time_steps) + 0.05 * np.cos(np.linspace(0, 2 * np.pi, time_steps))
        bandwidth = np.random.uniform(0.5, 2.0, size=time_steps)
        
        # More realistic multi-user interference model
        interference = 0.2 + 0.1 * np.random.rand(time_steps) + 0.05 * np.cos(np.linspace(0, 4 * np.pi, time_steps))
        interference = interference + 0.1 * np.sin(np.linspace(0, 4 * np.pi, time_steps))  # Correlated interference
        
        # Environmental effects: Weather-based attenuation and building shadowing
        rain_fade = 0.05 * np.random.binomial(1, 0.1, size=time_steps)  # Occasional rain fade
        building_shadowing = 0.1 * np.random.binomial(1, 0.05, size=time_steps)  # Random shadowing

        # User mobility with group movement pattern
        user_mobility = np.abs(np.random.normal(1.0, 0.3, size=time_steps))
        
        # Doppler shift with dynamic changes and random handover effects
        doppler_shift = 0.01 * user_mobility * signal_strength * (1 + 0.5 * np.random.normal(size=time_steps))
        
        # Composite multipath fading model with Nakagami distribution approximation
        multipath_fading = signal_strength * (1 + np.random.gamma(2, 0.5, size=time_steps))
        
        # Nonlinear SNR, with environmental and user-interference effects
        snr = np.clip((signal_strength / (noise_level + interference + rain_fade + building_shadowing)), 0, 5)
        
        # Periodic channel drops and adaptation to simulate real-world outages and power control
        outage = (np.random.rand(time_steps) < 0.1).astype(float)
        effective_signal = signal_strength * (1 - outage)  # Signal drops due to outage
        adaptive_power = effective_signal * (1 + 0.1 * np.sin(np.linspace(0, np.pi, time_steps)))  # Adaptive power control
        
        # Channel quality metric considering all effects
        channel_quality = np.mean(snr * multipath_fading - doppler_shift) * (1 - 0.1 * outage.mean())
        
        # Append data for each time step
        for t in range(time_steps):
            data.append([
                adaptive_power[t], noise_level[t], bandwidth[t], interference[t], rain_fade[t],
                building_shadowing[t], user_mobility[t], snr[t], doppler_shift[t],
                multipath_fading[t], outage[t], channel_quality
            ])
    
    columns = ['adaptive_power', 'noise_level', 'bandwidth', 'interference', 'rain_fade',
               'building_shadowing', 'user_mobility', 'snr', 'doppler_shift',
               'multipath_fading', 'outage', 'channel_quality']
    df = pd.DataFrame(data, columns=columns)
    
    df.to_csv('ultra_complex_wireless_channel_data.csv', index=False)
    print("Ultra-complex data saved to 'ultra_complex_wireless_channel_data.csv'")
    
    return df

df = generate_channel_data()
