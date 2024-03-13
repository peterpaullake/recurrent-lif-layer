from typing import Callable

import numpy as np
import torch

import cuda_spike_tools


class DerivativeEvaluator:
    """
    Generic derivative evaluator.
    """

    def __init__(
        self, leak_conductance=1.0, tau_membrane=2.0, tau_synapse=1.0,
        threshold_potential=1.0, reset_potential=0.5, refractory_period=0.2,
    ):
        self.leak_conductance = leak_conductance
        self.tau_membrane = tau_membrane
        self.tau_synapse = tau_synapse
        self.threshold_potential = threshold_potential
        self.reset_potential = reset_potential
        self.refractory_period = refractory_period

    def get_input_spike_time_derivatives(
        self, input_spike_times, input_spike_synapse_ids,
        output_spike_times, output_spike_neuron_ids,
        feedforward_weights, recurrent_weights,
    ):
        """
        input_spike_times       : (batch_size, max_input_spikes)
        input_spike_synapse_ids : (batch_size, max_input_spikes)
        output_spike_times      : (batch_size, max_output_spikes)
        output_spike_neuron_ids : (batch_size, max_output_spikes)
        feedforward_weights     : (n_neurons, n_synapses)
        recurrent_weights       : (n_neurons, n_neurons)
        returns                 : (batch_size, max_output_spikes, max_input_spikes)
        """

        raise NotImplementedError

    def get_feedforward_weight_derivatives(
        self, input_spike_times, input_spike_synapse_ids,
        output_spike_times, output_spike_neuron_ids,
        feedforward_weights, recurrent_weights,
    ):
        """
        input_spike_times       : (batch_size, max_input_spikes)
        input_spike_synapse_ids : (batch_size, max_input_spikes)
        output_spike_times      : (batch_size, max_output_spikes)
        output_spike_neuron_ids : (batch_size, max_output_spikes)
        feedforward_weights     : (n_neurons, n_synapses)
        recurrent_weights       : (n_neurons, n_neurons)
        returns                 : (batch_size, max_output_spikes, n_neurons, n_synapses)
        """

        raise NotImplementedError

    def get_recurrent_weight_derivatives(
        self, input_spike_times, input_spike_synapse_ids,
        output_spike_times, output_spike_neuron_ids,
        feedforward_weights, recurrent_weights,
    ):
        """
        input_spike_times       : (batch_size, max_input_spikes)
        input_spike_synapse_ids : (batch_size, max_input_spikes)
        output_spike_times      : (batch_size, max_output_spikes)
        output_spike_neuron_ids : (batch_size, max_output_spikes)
        feedforward_weights     : (n_neurons, n_synapses)
        recurrent_weights       : (n_neurons, n_neurons)
        returns                 : (batch_size, max_output_spikes, n_neurons, n_neurons)
        """

        raise NotImplementedError

    @staticmethod
    def _input_shapes_valid(
        input_spike_times, input_spike_synapse_ids,
        output_spike_times, output_spike_neuron_ids,
        feedforward_weights, recurrent_weights
    ):
        """
        Returns whether the given input tensors have the correct shapes.
        """

        return (
            len(input_spike_times.shape) == 2
            and len(input_spike_synapse_ids.shape) == 2
            and len(output_spike_times.shape) == 2
            and len(output_spike_neuron_ids.shape) == 2
            and len(feedforward_weights.shape) == 2
            and len(recurrent_weights.shape) == 2
            and input_spike_times.shape == input_spike_synapse_ids.shape
            and output_spike_times.shape == output_spike_neuron_ids.shape
            and input_spike_times.shape[0] == output_spike_times.shape[0]
            and (
                feedforward_weights.shape[0]
                == recurrent_weights.shape[0]
                == recurrent_weights.shape[1]
            )
        )


class CUDADerivativeEvaluator(DerivativeEvaluator):
    """
    Fast derivative evaluator using CUDA.
    """

    def get_input_spike_time_derivatives(
        self, input_spike_times, input_spike_synapse_ids,
        output_spike_times, output_spike_neuron_ids,
        feedforward_weights, recurrent_weights,
    ):
        assert self._input_shapes_valid(
            input_spike_times, input_spike_synapse_ids,
            output_spike_times, output_spike_neuron_ids,
            feedforward_weights, recurrent_weights,
        )
        assert self.__is_contiguous_cuda_tensor(input_spike_times, torch.float32)
        assert self.__is_contiguous_cuda_tensor(input_spike_synapse_ids, torch.int32)
        assert self.__is_contiguous_cuda_tensor(output_spike_times, torch.float32)
        assert self.__is_contiguous_cuda_tensor(output_spike_neuron_ids, torch.int32)
        assert self.__is_contiguous_cuda_tensor(feedforward_weights, torch.float32)
        assert self.__is_contiguous_cuda_tensor(recurrent_weights, torch.float32)

        return cuda_spike_tools.get_input_spike_time_derivatives(
            input_spike_times, input_spike_synapse_ids,
            output_spike_times, output_spike_neuron_ids,
            feedforward_weights, recurrent_weights,
            self.leak_conductance,
            self.tau_membrane,
            self.tau_synapse,
            self.threshold_potential,
            self.reset_potential,
            self.refractory_period,
        )

    def get_feedforward_weight_derivatives(
        self, input_spike_times, input_spike_synapse_ids,
        output_spike_times, output_spike_neuron_ids,
        feedforward_weights, recurrent_weights,
    ):
        assert self._input_shapes_valid(
            input_spike_times, input_spike_synapse_ids,
            output_spike_times, output_spike_neuron_ids,
            feedforward_weights, recurrent_weights,
        )
        assert self.__is_contiguous_cuda_tensor(input_spike_times, torch.float32)
        assert self.__is_contiguous_cuda_tensor(input_spike_synapse_ids, torch.int32)
        assert self.__is_contiguous_cuda_tensor(output_spike_times, torch.float32)
        assert self.__is_contiguous_cuda_tensor(output_spike_neuron_ids, torch.int32)
        assert self.__is_contiguous_cuda_tensor(feedforward_weights, torch.float32)
        assert self.__is_contiguous_cuda_tensor(recurrent_weights, torch.float32)

        return cuda_spike_tools.get_feedforward_weight_derivatives(
            input_spike_times, input_spike_synapse_ids,
            output_spike_times, output_spike_neuron_ids,
            feedforward_weights, recurrent_weights,
            self.leak_conductance,
            self.tau_membrane,
            self.tau_synapse,
            self.threshold_potential,
            self.reset_potential,
            self.refractory_period,
        )

    def get_recurrent_weight_derivatives(
        self, input_spike_times, input_spike_synapse_ids,
        output_spike_times, output_spike_neuron_ids,
        feedforward_weights, recurrent_weights,
    ):
        assert self._input_shapes_valid(
            input_spike_times, input_spike_synapse_ids,
            output_spike_times, output_spike_neuron_ids,
            feedforward_weights, recurrent_weights,
        )
        assert self.__is_contiguous_cuda_tensor(input_spike_times, torch.float32)
        assert self.__is_contiguous_cuda_tensor(input_spike_synapse_ids, torch.int32)
        assert self.__is_contiguous_cuda_tensor(output_spike_times, torch.float32)
        assert self.__is_contiguous_cuda_tensor(output_spike_neuron_ids, torch.int32)
        assert self.__is_contiguous_cuda_tensor(feedforward_weights, torch.float32)
        assert self.__is_contiguous_cuda_tensor(recurrent_weights, torch.float32)

        return cuda_spike_tools.get_recurrent_weight_derivatives(
            input_spike_times, input_spike_synapse_ids,
            output_spike_times, output_spike_neuron_ids,
            feedforward_weights, recurrent_weights,
            self.leak_conductance,
            self.tau_membrane,
            self.tau_synapse,
            self.threshold_potential,
            self.reset_potential,
            self.refractory_period,
        )

    @staticmethod
    def __is_contiguous_cuda_tensor(x, dtype):
        """
        Returns whether `x` is a C-contiguous PyTorch CUDA tensor of the given type.
        """

        return torch.is_tensor(x) and x.is_contiguous() and x.is_cuda and x.dtype is dtype


class FiniteDifferenceDerivativeEvaluator(DerivativeEvaluator):
    """
    Slow, simple, inexact derivative evaluator based on finite differences.
    Intended for debugging and comparison with `CUDADerivativeEvaluator`.
    """

    def __init__(self, spike_time_solver, epsilon=1e-3, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.spike_time_solver = spike_time_solver
        self.epsilon = epsilon

    @staticmethod
    def __devices_equal(*tensors):
        """
        Returns whether the given input tensors are all on the same device.
        """

        if len(tensors) == 0:
            return True
        else:
            return all([tensor.device == tensors[0].device for tensor in tensors[1:]])

    @staticmethod
    def __evaluate_finite_difference_derivative(
        f: Callable[[np.ndarray], np.ndarray], x: np.ndarray, epsilon: float
    ) -> np.ndarray:
        output_shape = f(x).shape
        input_shape = x.shape
        n_derivatives = input_shape[-1]

        derivatives = np.empty((*output_shape, n_derivatives))
        for derivative_id in range(n_derivatives):
            dx = np.zeros(n_derivatives)
            dx[derivative_id] = epsilon
            infs_to_zero = lambda array : np.where(np.isinf(array), 0, array)
            finite_differences = infs_to_zero(f(x + dx)) - infs_to_zero(f(x))
            derivatives[..., derivative_id] = finite_differences / epsilon

        return derivatives

    def get_input_spike_time_derivatives(
        self, input_spike_times, input_spike_synapse_ids,
        output_spike_times, output_spike_neuron_ids,
        feedforward_weights, recurrent_weights,
    ):
        assert self._input_shapes_valid(
            input_spike_times, input_spike_synapse_ids,
            output_spike_times, output_spike_neuron_ids,
            feedforward_weights, recurrent_weights,
        )
        assert self.__devices_equal(
            input_spike_times, input_spike_synapse_ids,
            output_spike_times, output_spike_neuron_ids,
            feedforward_weights, recurrent_weights,
        )

        max_output_spikes = output_spike_times.shape[1]
        device = input_spike_times.device

        def f(x):
            output_spike_times, output_spike_neuron_ids = self.spike_time_solver.get_output_spike_times(
                torch.FloatTensor(x).to(device),
                input_spike_synapse_ids,
                feedforward_weights,
                recurrent_weights,
                max_output_spikes,
            )
            return np.array(output_spike_times.cpu())

        return torch.FloatTensor(self.__evaluate_finite_difference_derivative(
            f, np.array(input_spike_times.cpu()), self.epsilon
        )).to(device)

    def get_feedforward_weight_derivatives(
        self, input_spike_times, input_spike_synapse_ids,
        output_spike_times, output_spike_neuron_ids,
        feedforward_weights, recurrent_weights,
    ):
        assert self._input_shapes_valid(
            input_spike_times, input_spike_synapse_ids,
            output_spike_times, output_spike_neuron_ids,
            feedforward_weights, recurrent_weights,
        )
        assert self.__devices_equal(
            input_spike_times, input_spike_synapse_ids,
            output_spike_times, output_spike_neuron_ids,
            feedforward_weights, recurrent_weights,
        )

        batch_size, max_output_spikes = output_spike_times.shape
        n_neurons, n_synapses = feedforward_weights.shape
        device = input_spike_times.device

        def f(x):
            output_spike_times, output_spike_neuron_ids = self.spike_time_solver.get_output_spike_times(
                input_spike_times,
                input_spike_synapse_ids,
                torch.FloatTensor(x.reshape(n_neurons, n_synapses)).to(device),
                recurrent_weights,
                max_output_spikes,
            )
            return np.array(output_spike_times.cpu())

        target_shape = (batch_size, max_output_spikes, n_neurons, n_synapses)
        return torch.FloatTensor(self.__evaluate_finite_difference_derivative(
            f, np.array(feedforward_weights.cpu()).flatten(), self.epsilon
        ).reshape(*target_shape)).to(device)

    def get_recurrent_weight_derivatives(
        self, input_spike_times, input_spike_synapse_ids,
        output_spike_times, output_spike_neuron_ids,
        feedforward_weights, recurrent_weights,
    ):
        assert self._input_shapes_valid(
            input_spike_times, input_spike_synapse_ids,
            output_spike_times, output_spike_neuron_ids,
            feedforward_weights, recurrent_weights,
        )
        assert self.__devices_equal(
            input_spike_times, input_spike_synapse_ids,
            output_spike_times, output_spike_neuron_ids,
            feedforward_weights, recurrent_weights,
        )

        batch_size, max_output_spikes = output_spike_times.shape
        n_neurons, _ = recurrent_weights.shape
        device = input_spike_times.device

        def f(x):
            output_spike_times, output_spike_neuron_ids = self.spike_time_solver.get_output_spike_times(
                input_spike_times,
                input_spike_synapse_ids,
                feedforward_weights,
                torch.FloatTensor(x.reshape(n_neurons, n_neurons)).to(device),
                max_output_spikes,
            )
            return np.array(output_spike_times.cpu())

        target_shape = (batch_size, max_output_spikes, n_neurons, n_neurons)
        return torch.FloatTensor(self.__evaluate_finite_difference_derivative(
            f, np.array(recurrent_weights.cpu()).flatten(), self.epsilon
        ).reshape(*target_shape)).to(device)
