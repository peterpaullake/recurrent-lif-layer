import unittest

import numpy as np
import torch

from recurrent_lif_layer.derivative_evaluator import CUDADerivativeEvaluator
from recurrent_lif_layer.derivative_evaluator import FiniteDifferenceDerivativeEvaluator
from recurrent_lif_layer.spike_time_solver import CUDASpikeTimeSolver


class TestCUDADerivativeEvaluator(unittest.TestCase):
    """
    Unit tests for the `CUDADerivativeEvaluator` class.
    """

    def setUp(self) -> None:
        # Define the problem shape
        batch_size = 32
        max_input_spikes = 8
        n_neurons = 4
        n_synapses = 3
        self.max_output_spikes = 9

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

    def test_derivative_methods(self) -> None:
        for method in [
            lambda derivative_evaluator : derivative_evaluator.get_input_spike_time_derivatives,
            lambda derivative_evaluator : derivative_evaluator.get_feedforward_weight_derivatives,
            lambda derivative_evaluator : derivative_evaluator.get_recurrent_weight_derivatives,
        ]:
            for use_recurrent_connections in [False, True]:
                if use_recurrent_connections:
                    recurrent_weights = self.recurrent_weights
                else:
                    recurrent_weights = torch.zeros_like(self.recurrent_weights)

                # Evaluate for the derivatives using the CUDA
                # evaluator and the finite difference evaluator
                cuda_spike_time_solver = CUDASpikeTimeSolver()
                output_spike_times, output_spike_neuron_ids = cuda_spike_time_solver.get_output_spike_times(
                    self.input_spike_times, self.input_spike_synapse_ids,
                    self.feedforward_weights, recurrent_weights,
                    self.max_output_spikes
                )
                cuda_derivatives = CUDADerivativeEvaluator().get_input_spike_time_derivatives(
                    self.input_spike_times, self.input_spike_synapse_ids,
                    output_spike_times, output_spike_neuron_ids,
                    self.feedforward_weights, recurrent_weights,
                )
                fd_derivatives = FiniteDifferenceDerivativeEvaluator(
                    cuda_spike_time_solver
                ).get_input_spike_time_derivatives(
                    self.input_spike_times, self.input_spike_synapse_ids,
                    output_spike_times, output_spike_neuron_ids,
                    self.feedforward_weights, recurrent_weights,
                )

                # Convert the results to numpy arrays
                cuda_derivatives = np.array(cuda_derivatives.cpu())
                fd_derivatives = np.array(fd_derivatives.cpu())

                # Compare the results from the two methods. We use a large
                # tolerance here since the finite difference method is inexact.
                self.assertTrue(np.abs(cuda_derivatives - fd_derivatives).max() < 5e-2)


if __name__ == "__main__":
    unittest.main()
