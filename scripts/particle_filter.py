import numpy as np
import matplotlib.pyplot as plt

# Define parameters
N = 500  # Number of particles
T = 100   # Number of time steps

# Define process model: x_k = f(x_{k-1}) + noise
def process_model(x):
    return 0.5 * x + 25 * x / (1 + x**2) + 8 * np.cos(1.2)

# Define measurement model: y_k = h(x_k) + noise
def measurement_model(x):
    return x**2 / 20

# Generate ground truth and measurements
true_states = [0]
measurements = []
for t in range(T):
    true_states.append(process_model(true_states[-1])+ np.random.normal(0, 0.5))
    measurements.append(measurement_model(true_states[-1]) + np.random.normal(0, 2))

# Particle Filter Implementation
particles = np.random.normal(0, 5, size=N)  # Initialize particles
weights = np.ones(N) / N                    # Initialize weights
estimated_states = []

for t in range(T):
    # Predict step
    particles = process_model(particles) + np.random.normal(0, 2, size=N)
    
    # Update step (weights)
    weights = np.exp(-0.5 * (measurements[t] - measurement_model(particles))**2)
    weights += 1e-300  # Avoid numerical issues
    weights /= np.sum(weights)
    
    # Resampling step
    particles = np.random.choice(particles, size=N, p=weights)
    weights = np.ones(N) / N  # Reset weights
    
    # Estimate state
    estimated_states.append(np.mean(particles))

# Plot results
plt.figure(figsize=(12, 6))
plt.plot(range(T), true_states[1:], label='True State', color='blue')
plt.plot(range(T), measurements, label='Measurements', color='green', linestyle='dotted')
plt.plot(range(T), estimated_states, label='Particle Filter Estimate', color='red')
plt.legend()
plt.xlabel('Time Step')
plt.ylabel('State')
plt.title('Particle Filter Results')
plt.show()
