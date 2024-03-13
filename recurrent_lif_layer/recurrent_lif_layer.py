import torch
import torch.nn as nn


class RecurrentLIFLayerFunction(torch.autograd.Function):
    """
    Custom `autograd.Function` class for calculating the forward and backward
    pass of a fully connected layer of LIF neurons with recurrent connections.
    
    https://pytorch.org/docs/stable/notes/extending.html
    """

    @staticmethod
    def forward(
        ctx,
        input_spike_times,
        input_spike_synapse_ids,
        feedforward_weights,
        recurrent_weights,
        max_output_spikes,
        spike_time_solver,
        derivative_evaluator,
        enable_recurrence,
    ):
        """
        input_spike_times       : (batch_size, max_input_spikes)
        input_spike_synapse_ids : (batch_size, max_input_spikes)
        feedforward_weights     : (n_neurons, n_synapses)
        recurrent_weights       : (n_neurons, n_neurons)
        returns                 : (batch_size, max_output_spikes),
                                  (batch_size, max_output_spikes)
        """

        assert len(input_spike_times.shape) == 2
        assert len(input_spike_synapse_ids.shape) == 2
        assert len(feedforward_weights.shape) == 2
        assert len(recurrent_weights.shape) == 2
        assert input_spike_times.shape == input_spike_synapse_ids.shape
        assert feedforward_weights.shape[0] == recurrent_weights.shape[0]
        assert recurrent_weights.shape[0] == recurrent_weights.shape[1]

        assert not feedforward_weights.isnan().any()
        assert not recurrent_weights.isnan().any()

        output_spike_times, output_spike_neuron_ids = spike_time_solver.get_output_spike_times(
            input_spike_times,
            input_spike_synapse_ids,
            feedforward_weights,
            recurrent_weights,
            max_output_spikes,
        )

        # Save the tensors needed to calculate the backward
        # pass together with the derivative evaluator
        ctx.save_for_backward(
            input_spike_times, input_spike_synapse_ids,
            output_spike_times, output_spike_neuron_ids,
            feedforward_weights, recurrent_weights,
        )
        ctx.derivative_evaluator = derivative_evaluator
        ctx.enable_recurrence = enable_recurrence

        return output_spike_times, output_spike_neuron_ids

    @staticmethod
    def backward(ctx, grad_output, _):
        """
        grad_output : (batch_size, max_output_spikes)
        returns     : (batch_size, max_input_spikes),
                      None,
                      (n_neurons, n_synapses),
                      (n_neurons, n_neurons),
                      None,
                      None,
                      None,
                      None
        """

        # (batch_size, max_output_spikes, max_input_spikes)
        input_spike_time_derivatives = ctx.derivative_evaluator.get_input_spike_time_derivatives(
            *ctx.saved_tensors
        )

        # (batch_size, max_output_spikes, n_neurons, n_synapses)
        feedforward_weight_derivatives = ctx.derivative_evaluator.get_feedforward_weight_derivatives(
            *ctx.saved_tensors
        )

        if ctx.enable_recurrence:
            # (batch_size, max_output_spikes, n_neurons, n_neurons)
            recurrent_weight_derivatives = ctx.derivative_evaluator.get_recurrent_weight_derivatives(
                *ctx.saved_tensors
            )

        # (batch_size, max_input_spikes)
        input_spike_times_gradient = torch.einsum('bo,boi->bi', grad_output, input_spike_time_derivatives)

        # (n_neurons, n_synapses)
        feedforward_weights_gradient = torch.einsum('bo,bons->ns', grad_output, feedforward_weight_derivatives)

        if ctx.enable_recurrence:
            # (n_neurons, n_neurons)
            recurrent_weights_gradient = torch.einsum('bo,bots->ts', grad_output, recurrent_weight_derivatives)
        else:
            n_neurons = len(feedforward_weights_gradient)
            recurrent_weights_gradient = torch.zeros((n_neurons, n_neurons), device=grad_output.device)

        return (
            input_spike_times_gradient,
            None,
            feedforward_weights_gradient,
            recurrent_weights_gradient,
            None, None, None, None,
        )


class RecurrentLIFLayer(nn.Module):
    """
    PyTorch module implementing a fully connected layer of LIF neurons with recurrent connections.
    """

    def __init__(
        self,
        n_neurons,
        n_synapses,
        spike_time_solver_class,
        derivative_evaluator_class,
        enable_multispike=True,
        enable_recurrence=True,
    ):
        super().__init__()

        self.feedforward_weights = nn.Parameter(torch.randn(n_neurons, n_synapses) + 1.0)
        self.recurrent_weights = nn.Parameter(torch.randn(n_neurons, n_neurons) + 1.0)

        if enable_multispike:
            self.spike_time_solver = spike_time_solver_class()
            self.derivative_evaluator = derivative_evaluator_class()
        else:
            # XXX avoiding multiple spikes using a long refractory period
            # is very hacky. Instead, we should pass the `enable_multispike`
            # flag to the solver and have it change its behavior accordingly.
            long_refractory_period = 1000.0
            self.spike_time_solver = spike_time_solver_class(refractory_period=long_refractory_period)
            self.derivative_evaluator = derivative_evaluator_class(refractory_period=long_refractory_period)
        self.enable_multispike = enable_multispike
        self.enable_recurrence = enable_recurrence

    def forward(self, input_spike_times, input_spike_synapse_ids, max_output_spikes=None):
        """
        input_spike_times       : (batch_size, max_input_spikes)
        input_spike_synapse_ids : (batch_size, max_input_spikes)
        returns                 : (batch_size, max_output_spikes),
                                  (batch_size, max_output_spikes)
        """

        assert len(input_spike_times.shape) == 2
        assert input_spike_times.shape == input_spike_synapse_ids.shape

        if max_output_spikes is None:
            max_input_spikes = input_spike_times.shape[1]
            max_output_spikes = max_input_spikes

        output_spike_times, output_spike_neuron_ids = RecurrentLIFLayerFunction.apply(
            input_spike_times,
            input_spike_synapse_ids,
            self.feedforward_weights,
            self.recurrent_weights,
            max_output_spikes,
            self.spike_time_solver,
            self.derivative_evaluator,
            self.enable_recurrence,
        )

        # If multi-spike is disabled, make sure no neuron fires more than once
        if not self.enable_multispike:
            n_spikes = self.calculate_n_spikes(output_spike_times, output_spike_neuron_ids, n_neurons=len(self))
            if (n_spikes > 1).any():
                raise RuntimeError('A neuron fired more than once despite multi-spike being disabled')

        return output_spike_times, output_spike_neuron_ids

    @staticmethod
    def calculate_n_spikes(output_spike_times, output_spike_neuron_ids, n_neurons):
        """
        Utility function for calculating how many times each neuron fired for each example in a batch.
        
        output_spike_times      : (batch_size, max_output_spikes)
        output_spike_neuron_ids : (batch_size, max_output_spikes)
        returns                 : (batch_size, n_neurons)
        """

        # Mark infinite output spikes with -1
        output_spike_neuron_ids = output_spike_neuron_ids.clone()
        output_spike_neuron_ids[output_spike_times.isinf()] = -1

        # (batch_size, max_output_spikes, n_neurons)
        n_spikes = (
            output_spike_neuron_ids.unsqueeze(-1)
            == torch.arange(n_neurons, device=output_spike_neuron_ids.device)
        )

        # (batch_size, n_neurons)
        return n_spikes.sum(dim=1)

    def __len__(self):
        """
        Returns the number of neurons in this layer.
        """

        return len(self.feedforward_weights)
