from typing import Callable, Tuple

import numpy as np
import torch
from tqdm import tqdm

import cuda_spike_tools


class SpikeTimeSolver:
    """
    Generic spike time solver.
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

    def get_output_spike_times(
        self, input_spike_times, input_spike_synapse_ids,
        feedforward_weights, recurrent_weights, max_output_spikes,
    ):
        """
        input_spike_times       : (batch_size, max_input_spikes)
        input_spike_synapse_ids : (batch_size, max_input_spikes)
        feedforward_weights     : (n_neurons, n_synapses)
        recurrent_weights       : (n_neurons, n_neurons)
        returns                 : (batch_size, max_output_spikes),
                                  (batch_size, max_output_spikes)
        """

        raise NotImplementedError

    @staticmethod
    def _input_shapes_valid(
        input_spike_times, input_spike_synapse_ids, feedforward_weights, recurrent_weights
    ):
        """
        Returns whether the given input tensors have the correct shapes.
        """

        return (
            len(input_spike_times.shape) == 2
            and len(input_spike_synapse_ids.shape) == 2
            and len(feedforward_weights.shape) == 2
            and len(recurrent_weights.shape) == 2
            and input_spike_times.shape == input_spike_synapse_ids.shape
            and (
                feedforward_weights.shape[0]
                == recurrent_weights.shape[0]
                == recurrent_weights.shape[1]
            )
        )


class CUDASpikeTimeSolver(SpikeTimeSolver):
    """
    Fast spike time solver using CUDA.
    """

    def get_output_spike_times(
        self, input_spike_times, input_spike_synapse_ids,
        feedforward_weights, recurrent_weights, max_output_spikes,
    ):
        assert self._input_shapes_valid(
            input_spike_times, input_spike_synapse_ids, feedforward_weights, recurrent_weights
        )
        assert self.__is_contiguous_cuda_tensor(input_spike_times, torch.float32)
        assert self.__is_contiguous_cuda_tensor(input_spike_synapse_ids, torch.int32)
        assert self.__is_contiguous_cuda_tensor(feedforward_weights, torch.float32)
        assert self.__is_contiguous_cuda_tensor(recurrent_weights, torch.float32)

        output_spike_times, output_spike_neuron_ids = cuda_spike_tools.get_output_spike_times(
            input_spike_times,
            input_spike_synapse_ids,
            feedforward_weights,
            recurrent_weights,
            max_output_spikes,
            self.leak_conductance,
            self.tau_membrane,
            self.tau_synapse,
            self.threshold_potential,
            self.reset_potential,
            self.refractory_period,
        )
        return output_spike_times, output_spike_neuron_ids

    @staticmethod
    def __is_contiguous_cuda_tensor(x, dtype):
        """
        Returns whether `x` is a C-contiguous PyTorch CUDA tensor of the given type.
        """

        return torch.is_tensor(x) and x.is_contiguous() and x.is_cuda and x.dtype is dtype


class RungeKuttaSpikeTimeSolver(SpikeTimeSolver):
    """
    Slow, simple spike time solver using Runge-Kutta integration.
    Intended for debugging and comparison with `CUDASpikeTimeSolver`.
    """

    @staticmethod
    def __integrate_runge_kutta(
        dydt: Callable[[float, np.ndarray], np.ndarray],
        t0: float,
        y0: np.ndarray,
        h: float,
        stop_condition: Callable,
        mutate_y: Callable = lambda t, y, h : None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        assert len(y0.shape) == 1
        n_dimensions = len(y0)

        times = np.array([t0])
        states = np.empty((1, n_dimensions))
        states[0] = y0

        t = t0
        y = y0
        while not stop_condition(t, y):
            # Integrate through one timestep
            k1 = dydt(t, y)
            k2 = dydt(t + h / 2, y + h * k1 / 2)
            k3 = dydt(t + h / 2, y + h * k2 / 2)
            k4 = dydt(t + h, y + h * k3)
            y += h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            t += h

            # Optionally mutate the state
            mutate_y(t, y, h)

            # Store the new state
            times = np.append(times, t)
            states = np.vstack((states, y))

        return times, states

    def __calculate_synaptic_current(self, t, spike_times, weight_ids, weights):
        """
        Calculates the synaptic current at the time `t` due to the
        given arrays of spike times and corresponding weight values.

        spike_times : (n_spike_times,)
        weight_ids  : (n_spike_times,)
        weights     : (n_neurons, n_weights)
        returns     : (n_neurons,)
        """

        # (1, n_spike_times)
        spike_time_differences = t - spike_times
        spike_time_differences = spike_time_differences.reshape(1, -1)

        # Here we set all negative spike time differences to -1.0. This
        # has no effect on the calculation since negative spike time
        # differences are zeroed out anyway by the Heaviside function
        # below. We do this just to silence warnings about infs.
        spike_time_differences[spike_time_differences < 0] = -1.0

        # (n_neurons, n_spike_times)
        spike_weights = weights[:, weight_ids]

        # (n_neurons, n_spike_times)
        terms = (
            spike_weights
            * np.heaviside(spike_time_differences, 1.0)
            * np.exp(-spike_time_differences / self.tau_synapse)
        )

        # (n_neurons,)
        return terms.sum(axis=1)

    def get_output_spike_times(
        self, input_spike_times, input_spike_synapse_ids, feedforward_weights,
        recurrent_weights, max_output_spikes, return_membrane_traces=False,
    ):
        assert self._input_shapes_valid(
            input_spike_times, input_spike_synapse_ids, feedforward_weights, recurrent_weights
        )
        assert (
            input_spike_times.device
            == input_spike_synapse_ids.device
            == feedforward_weights.device
            == recurrent_weights.device
        )

        device = input_spike_times.device
        batch_size, _ = input_spike_times.shape
        n_neurons, _ = feedforward_weights.shape

        input_spike_times = np.array(input_spike_times.cpu())
        input_spike_synapse_ids = np.array(input_spike_synapse_ids.cpu())
        feedforward_weights = np.array(feedforward_weights.cpu())
        recurrent_weights = np.array(recurrent_weights.cpu())

        if batch_size > 1:
            output_spike_times = torch.empty((batch_size, max_output_spikes))
            output_spike_neuron_ids = torch.empty((batch_size, max_output_spikes), dtype=torch.int32)
            for example_id in tqdm(range(batch_size)):
                example_output_spike_times, example_output_neuron_ids = self.get_output_spike_times(
                    torch.FloatTensor(input_spike_times[example_id].reshape(1, -1)),
                    torch.IntTensor(input_spike_synapse_ids[example_id].reshape(1, -1)),
                    torch.FloatTensor(feedforward_weights),
                    torch.FloatTensor(recurrent_weights),
                    max_output_spikes,
                )
                output_spike_times[example_id] = example_output_spike_times[0]
                output_spike_neuron_ids[example_id] = example_output_neuron_ids[0]
            return output_spike_times.to(device), output_spike_neuron_ids.to(device)
        else:
            input_spike_times = input_spike_times[0]
            input_spike_synapse_ids = input_spike_synapse_ids[0]

        # Set up arrays to store the output spikes
        output_spike_times = np.full(max_output_spikes, np.inf)
        output_spike_neuron_ids = np.empty(max_output_spikes, dtype=np.int32)
        n_output_spikes = 0

        # Set up arrays to store the refractory status of the neurons
        is_refractory = np.zeros(n_neurons, dtype=bool)
        refractory_period_remaining = np.empty(n_neurons)

        def dydt(t, membrane_potentials):
            leak_term = -membrane_potentials / self.tau_membrane

            feedforward_current = self.__calculate_synaptic_current(
                t,
                input_spike_times,
                input_spike_synapse_ids,
                feedforward_weights,
            )
            recurrent_current = self.__calculate_synaptic_current(
                t,
                output_spike_times[:n_output_spikes],
                output_spike_neuron_ids[:n_output_spikes],
                recurrent_weights,
            )
            synapse_term = (
                (feedforward_current + recurrent_current)
                / (self.leak_conductance * self.tau_membrane)
            )

            return np.where(is_refractory, 0.0, leak_term + synapse_term)

        def mutate_membrane_potentials(t, membrane_potentials, h):
            """
            Detects threshold crossings and deals with refractory periods.
            """

            nonlocal output_spike_times
            nonlocal output_spike_neuron_ids
            nonlocal n_output_spikes
            nonlocal is_refractory
            nonlocal refractory_period_remaining

            # Hold refractory neurons at the reset potential
            membrane_potentials[is_refractory] = self.reset_potential

            # Detect whether any neurons have spiked
            is_spiked = np.logical_and(
                ~is_refractory, membrane_potentials >= self.threshold_potential
            )
            n_spiked_neurons = is_spiked.sum()

            # Record the output spikes
            for neuron_id in np.arange(n_neurons)[is_spiked]:
                output_spike_times[n_output_spikes] = t
                output_spike_neuron_ids[n_output_spikes] = neuron_id
                n_output_spikes += 1

            # Update the refractory status of the spiked neurons. Here we round the
            # refractory period to the nearest integer multiple of the timestep `h`.
            membrane_potentials[is_spiked] = self.threshold_potential
            is_refractory[is_spiked] = True
            refractory_period_remaining[is_spiked] = round(self.refractory_period / h) * h

            # Update the refractory period countdowns
            refractory_period_remaining[is_refractory] -= h
            refractory_period_elapsed = refractory_period_remaining <= 0
            is_refractory[np.logical_and(is_refractory, refractory_period_elapsed)] = False

        def stop_condition(t, membrane_potentials):
            return (
                n_output_spikes >= max_output_spikes
                or (
                    (t > input_spike_times[np.isfinite(input_spike_times)]).all()
                    and np.isclose(membrane_potentials, 0).all()
                )
            )

        times, membrane_potentials = self.__integrate_runge_kutta(
            dydt,
            t0=input_spike_times.min(),
            y0=np.zeros(n_neurons),
            h=self.tau_synapse/5000.0,
            stop_condition=stop_condition,
            mutate_y=mutate_membrane_potentials,
        )

        output_spike_times = torch.FloatTensor(output_spike_times.reshape(1, -1)).to(device)
        output_spike_neuron_ids = torch.IntTensor(output_spike_neuron_ids.reshape(1, -1)).to(device)
        if return_membrane_traces:
            return output_spike_times, output_spike_neuron_ids, times, membrane_potentials
        else:
            return output_spike_times, output_spike_neuron_ids
