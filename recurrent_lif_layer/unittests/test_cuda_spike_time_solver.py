import unittest

import numpy as np
import torch

from recurrent_lif_layer.spike_time_solver import CUDASpikeTimeSolver
from recurrent_lif_layer.spike_time_solver import RungeKuttaSpikeTimeSolver


class TestCUDASpikeTimeSolver(unittest.TestCase):
    """
    Unit tests for the `CUDASpikeTimeSolver` class.
    """

    def setUp(self) -> None:
        # Define the problem shape
        batch_size = 3
        max_input_spikes = 8
        n_neurons = 4
        n_synapses = 3
        self.max_output_spikes = 32

        # Generate random input arrays
        np.random.seed(0)
        device = torch.device('cuda')
        self.input_spike_times = torch.FloatTensor(
            np.random.uniform(0, 10, size=(batch_size, max_input_spikes))
        ).to(device)
        self.input_spike_synapse_ids = torch.IntTensor(
            np.random.randint(0, n_synapses, size=(batch_size, max_input_spikes))
        ).to(device)
        spike_counts = np.random.randint(0, max_input_spikes + 1, size=batch_size)
        for example_id, spike_count in enumerate(spike_counts):
            self.input_spike_times[example_id][np.arange(max_input_spikes) >= spike_count] = np.inf
        self.feedforward_weights = torch.FloatTensor(np.random.randn(n_neurons, n_synapses) + 2.3).to(device)
        self.recurrent_weights = torch.FloatTensor(np.random.randn(n_neurons, n_neurons) + 0.5).to(device)

    def test_get_output_spike_times(self) -> None:
        # Solve for the output spike times using the CUDA solver and the Runge-Kutta solver
        cuda_output_spike_times, cuda_output_spike_neuron_ids = CUDASpikeTimeSolver().get_output_spike_times(
            self.input_spike_times,
            self.input_spike_synapse_ids,
            self.feedforward_weights,
            self.recurrent_weights,
            self.max_output_spikes,
        )
        rk_output_spike_times, rk_output_spike_neuron_ids = RungeKuttaSpikeTimeSolver().get_output_spike_times(
            self.input_spike_times,
            self.input_spike_synapse_ids,
            self.feedforward_weights,
            self.recurrent_weights,
            self.max_output_spikes,
        )

        # Convert the results to numpy arrays
        cuda_output_spike_times = np.array(cuda_output_spike_times.cpu())
        cuda_output_spike_neuron_ids = np.array(cuda_output_spike_neuron_ids.cpu())

        rk_output_spike_times = np.array(rk_output_spike_times.cpu())
        rk_output_spike_neuron_ids = np.array(rk_output_spike_neuron_ids.cpu())

        # Replace infinite spike times and their corresponding neuron
        # ids with constant padding values to facilitate comparison
        cuda_is_inf = np.isinf(cuda_output_spike_times)
        cuda_output_spike_times[cuda_is_inf] = 0.0
        cuda_output_spike_neuron_ids[cuda_is_inf] = -1

        rk_is_inf = np.isinf(rk_output_spike_times)
        rk_output_spike_times[rk_is_inf] = 0.0
        rk_output_spike_neuron_ids[rk_is_inf] = -1

        # Compare the results from the two methods
        self.assertTrue(np.abs(cuda_output_spike_times - rk_output_spike_times).max() < 1e-3)
        self.assertTrue((cuda_output_spike_neuron_ids == rk_output_spike_neuron_ids).all())


if __name__ == "__main__":
    unittest.main()
