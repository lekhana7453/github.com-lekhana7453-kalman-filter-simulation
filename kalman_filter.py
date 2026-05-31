import numpy as np
import matplotlib.pyplot as plt

# --- Kalman Filter Parameters ---
dt = 0.1          # time step
time = np.arange(0, 10, dt)  # 10 seconds

# --- True Position (what we want to track) ---
true_position = np.sin(time) * 50 + 50  # smooth sine wave motion

# --- Noisy GPS Sensor Data ---
gps_noise = np.random.normal(0, 10, len(time))  # GPS error ±10 units
gps_measurements = true_position + gps_noise

# --- Kalman Filter Setup ---
x = 0.0       # initial position estimate
P = 1.0       # initial uncertainty

Q = 0.1       # process noise (how much we trust the model)
R = 10.0      # measurement noise (how noisy is GPS)

kalman_estimates = []

# --- Kalman Filter Loop ---
for z in gps_measurements:

    # PREDICT step
    x_pred = x
    P_pred = P + Q

    # UPDATE step
    K = P_pred / (P_pred + R)      # Kalman Gain
    x = x_pred + K * (z - x_pred)  # Update estimate
    P = (1 - K) * P_pred           # Update uncertainty

    kalman_estimates.append(x)

# --- Plot Results ---
plt.figure(figsize=(12, 6))
plt.plot(time, true_position, 'g-', linewidth=2, label='True Position')
plt.plot(time, gps_measurements, 'r.', alpha=0.4, markersize=5, label='Noisy GPS')
plt.plot(time, kalman_estimates, 'b-', linewidth=2, label='Kalman Filter Estimate')
plt.title('Kalman Filter - Sensor Fusion Simulation')
plt.xlabel('Time (seconds)')
plt.ylabel('Position')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()