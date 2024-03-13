import torch
import warnings

from recurrent_lif_layer import RecurrentLIFLayer
from recurrent_lif_layer.derivative_evaluator import CUDADerivativeEvaluator
from recurrent_lif_layer.spike_time_solver import CUDASpikeTimeSolver


class SNNClassifier(torch.nn.Module):
    """
    PyTorch module implementing a Fast and Deep-style
    hierarchical network composed of two recurrent LIF layers.
    
    input_size: The number of neurons in the input layer.
    hidden_size: The number of neurons in the hidden layer.
    max_hidden_output_spikes: The maximum allowed number of
        output spikes that can be returned by the hidden layer.
    n_classes: The number of neurons in the label layer.
    enable_multispike: Whether to allow multiple spikes per hidden neuron.
    enable_recurrence: Whether to feed hidden layer
        output spikes back into the hidden layer.
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        max_hidden_output_spikes,
        n_classes,
        enable_multispike=True,
        enable_recurrence=True,
    ):
        super().__init__()

        if hidden_size > max_hidden_output_spikes:
            warnings.warn(
                f'The number of hidden neurons ({hidden_size}) is greater '
                'than the maximum allowed number of hidden layer output spikes '
                f'({max_hidden_output_spikes}). This implies that, in response '
                'to any given example, not all hidden neurons can fire.'
            )

        self.n_classes = n_classes
        self.max_hidden_output_spikes = max_hidden_output_spikes
        hidden_layer = RecurrentLIFLayer(
            n_neurons=hidden_size,
            n_synapses=input_size,
            spike_time_solver_class=CUDASpikeTimeSolver,
            derivative_evaluator_class=CUDADerivativeEvaluator,
            enable_multispike=enable_multispike,
            enable_recurrence=enable_recurrence, 
        )
        output_layer = RecurrentLIFLayer(
            n_neurons=n_classes,
            n_synapses=hidden_size,
            spike_time_solver_class=CUDASpikeTimeSolver,
            derivative_evaluator_class=CUDADerivativeEvaluator,
            enable_multispike=False,
            enable_recurrence=enable_recurrence,
        )
        self.layers = torch.nn.ModuleList([hidden_layer, output_layer])

    @staticmethod
    def __sort_output_spikes(output_spike_times, output_spike_neuron_ids):
        """
        Returns indices that sort the output spikes by neuron id. This is
        used to convert the raw, chronologically sorted output spike times
        to spike times sorted by class id, which are suitable for prediction.

        Notes: For any given example, each neuron must fire no more than once.

        output_spike_times      : (batch_size, n_classes)
        output_spike_neuron_ids : (batch_size, n_classes)
        returns                 : (batch_size, n_classes)
        """

        assert len(output_spike_times.shape) == 2
        assert output_spike_times.shape == output_spike_neuron_ids.shape
        batch_size, n_classes = output_spike_times.shape

        # Make sure no neuron fires more than once
        n_spikes = RecurrentLIFLayer.calculate_n_spikes(
            output_spike_times, output_spike_neuron_ids, n_neurons=n_classes
        )
        if (n_spikes > 1).any():
            raise RuntimeError('A neuron in the output layer fired more than once')

        # Mark infinite output spikes with -1
        output_spike_neuron_ids = output_spike_neuron_ids.clone()
        output_spike_neuron_ids[output_spike_times.isinf()] = -1

        for example_id in range(batch_size):
            neuron_ids = output_spike_neuron_ids[example_id]
            non_inf_neuron_ids = neuron_ids[neuron_ids != -1]

            # Fill in the neuron ids of non-spiking neurons
            inf_neuron_ids = torch.IntTensor(
                sorted(set(range(n_classes)) - set(non_inf_neuron_ids.cpu().numpy())),
            ).to(output_spike_neuron_ids.device)
            output_spike_neuron_ids[example_id][neuron_ids == -1] = inf_neuron_ids

        return output_spike_neuron_ids.argsort(dim=1)

    def forward(self, x, return_all_layer_outputs=False):
        """
        x       : (batch_size, input_size)
        returns : (batch_size, n_classes)
        """

        batch_size, input_size = x.shape

        # Convert the input examples to an input spike train
        input_spike_times = x
        input_spike_synapse_ids = torch.stack([
            torch.arange(input_size, dtype=torch.int32) for _ in range(batch_size)
        ]).to(input_spike_times.device)

        # Evaluate the hidden and output layers
        if return_all_layer_outputs:
            layer_outputs = []
        spike_times, spike_neuron_ids = input_spike_times, input_spike_synapse_ids
        for layer_id, layer in enumerate(self.layers):
            is_hidden = layer_id < len(self.layers) - 1
            max_output_spikes = self.max_hidden_output_spikes if is_hidden else self.n_classes
            spike_times, spike_neuron_ids = layer(spike_times, spike_neuron_ids, max_output_spikes)
            if return_all_layer_outputs:
                layer_outputs.append((spike_times, spike_neuron_ids))
        output_spike_times, output_spike_neuron_ids = spike_times, spike_neuron_ids

        # Sort the output spike times by neuron id
        sorted_output_spike_times = torch.gather(
            output_spike_times, dim=1,
            index=self.__sort_output_spikes(
                output_spike_times, output_spike_neuron_ids
            ),
        )

        if return_all_layer_outputs:
            return sorted_output_spike_times, layer_outputs
        else:
            return sorted_output_spike_times


class TTFSLoss(torch.autograd.Function):
    """
    Custom PyTorch function for calculating the forward and
    backward pass of the TTFS loss. Gradient components associated
    with infinite label neuron spike times are set to zero.
    """

    @staticmethod
    def forward(ctx, label_spike_times, labels, xi=0.2, tau_synapse=1.0):
        """
        label_spike_times : (batch_size, n_classes)
        labels            : (batch_size)
        returns           : (batch_size)
        """

        assert len(label_spike_times.shape) == 2
        assert len(labels.shape) == 1
        assert label_spike_times.shape[0] == labels.shape[0]

        ctx.batch_size, ctx.n_classes = label_spike_times.shape
        ctx.xi = xi
        ctx.tau_synapse = tau_synapse

        # (batch_size)
        correct_label_spike_times = label_spike_times[
            torch.arange(ctx.batch_size), labels
        ]

        # (batch_size, 1)
        correct_label_spike_times = correct_label_spike_times.reshape(-1, 1)

        # (batch_size, n_classes)
        label_spike_time_differences = torch.where(
            torch.logical_and(
                label_spike_times.isinf(), correct_label_spike_times.isinf()
            ),
            0.0,
            label_spike_times - correct_label_spike_times,
        )

        # (batch_size)
        losses = torch.log(torch.sum(torch.exp(
            -label_spike_time_differences / (ctx.xi * ctx.tau_synapse)
        ), dim=1))

        ctx.save_for_backward(label_spike_time_differences, labels)
        return losses

    @staticmethod
    def backward(ctx, grad_output):
        """
        grad_output : (batch_size)
        returns     : (batch_size, n_classes)
        """

        label_spike_time_differences, labels, = ctx.saved_tensors

        # (batch_size, n_classes)
        exponential_terms = torch.exp(
            -label_spike_time_differences / (ctx.xi * ctx.tau_synapse)
        )

        # (batch_size, n_classes)
        term1 = exponential_terms

        # (batch_size)
        sums = exponential_terms.sum(dim=1)

        # (batch_size, n_classes)
        term2 = torch.zeros(ctx.batch_size, ctx.n_classes).to(grad_output.device)
        term2[torch.arange(ctx.batch_size), labels] = sums

        prefactor = -1 / (ctx.xi * ctx.tau_synapse)

        # (batch_size, n_classes)
        derivatives = prefactor * (term1 - term2) / sums.reshape(-1, 1)
        derivatives[derivatives.isnan()] = 0

        return grad_output.reshape(-1, 1) * derivatives, None, None, None


class WeightBumping:
    """
    Fast and Deep-style weight bumping for use during training to
    encourage more activity in the network when there are too many non-
    spiking neurons. This is useful because, since the loss gradient can't
    flow through non-spiking neurons, too many of them can hinder learning.
    https://github.com/JulianGoeltz/fastAndDeep/blob/main/src/training.py
    """

    @staticmethod
    def __find_silent_neurons(output_spike_times, output_spike_neuron_ids, n_neurons):
        """
        output_spike_times      : (batch_size, max_output_spikes)
        output_spike_neuron_ids : (batch_size, max_output_spikes)
        returns                 : (batch_size, n_neurons)
        """

        assert len(output_spike_times.shape) == 2
        batch_size, max_output_spikes = output_spike_times.shape
        assert output_spike_neuron_ids.shape == (batch_size, max_output_spikes)

        # Mark infinite output spikes with -1
        output_spike_neuron_ids = output_spike_neuron_ids.clone()
        output_spike_neuron_ids[output_spike_times.isinf()] = -1

        # Calculate for each example whether each neuron is silent
        # (batch_size, n_neurons)
        is_silent = self.calculate_n_spikes(output_spike_times, output_spike_neuron_ids, n_neurons) == 0
        return is_silent

    @classmethod
    def check(cls, layer_outputs, layer_sizes, prev_result=None, max_proportion_silent_neurons=0.3, initial_bump_size=5e-4):
        """
        Given the neuronal activity of all layers in a network in response to
        a batch of examples, returns the index of the shallowest layer that
        requires weight bumping together with the recommended bump size
        according to the exponential weight bumping scheme from Fast and Deep.
        https://github.com/JulianGoeltz/fastAndDeep/blob/main/src/training.py

        layer_outputs: List of pairs of output spike times and corresponding
            output spike neuron ids, with one such pair per layer.
        layer_sizes: The number of neurons in each layer.
        prev_result: The return value of the previous call to this function. Used to
            exponentially increase the bump size until the weight bumping has an effect.
        """

        assert len(layer_outputs) == len(layer_sizes)

        bump_layer_id = None
        for layer_id, ((output_spike_times, output_spike_neuron_ids), n_neurons) in enumerate(
            zip(layer_outputs, layer_sizes)
        ):
            # (batch_size, n_neurons)
            is_silent = cls.__find_silent_neurons(output_spike_times, output_spike_neuron_ids, n_neurons)

            # Decide whether to bump this layer based on the proportion of silent neurons
            silent_proportion = is_silent.float().mean()
            if silent_proportion > max_proportion_silent_neurons:
                bump_layer_id = layer_id
                break

        # Return if no layer requires bumping
        if bump_layer_id is None:
            return None

        if prev_result is None:
            bump_size = initial_bump_size
        else:
            # Extract the previously used bump settings
            prev_bump_layer_id, prev_bump_size = prev_result

            # If this same layer required bumping last time too, double the bump size
            if bump_layer_id == prev_bump_layer_id:
                bump_size = 2.0 * prev_bump_size
            else:
                bump_size = initial_bump_size

        return bump_layer_id, bump_size

    @classmethod
    def apply(cls, bump_size, feedforward_weights, recurrent_weights, output_spike_times, output_spike_neuron_ids):
        """
        Increments the weights connected to non-spiking neurons.

        feedforward_weights     : (n_neurons, n_synapses)
        recurrent_weights       : (n_neurons, n_neurons)
        output_spike_times      : (batch_size, max_output_spikes)
        output_spike_neuron_ids : (batch_size, max_output_spikes)
        """

        n_neurons = len(feedforward_weights)

        # (batch_size, n_neurons)
        is_silent = cls.__find_silent_neurons(output_spike_times, output_spike_neuron_ids, n_neurons)

        # Calculate which neurons are silent for at least one example
        # (n_neurons)
        is_silent = is_silent.sum(axis=0).bool()

        # Apply the bumps to only the silent neurons
        weight_bumps = bump_size * is_silent.reshape(n_neurons, 1)
        feedforward_weights.data += weight_bumps
        recurrent_weights.data += weight_bumps


def calculate_average_statistic(model, dataloader, device, calculate_statistic=lambda outputs, layer_outputs, labels: 0.0):
    """
    Returns the result of `calculate_statistic` averaged over the entire dataset.
    """

    statistic_sum = 0.0
    n_examples = 0

    model.eval()
    for features, labels in dataloader:
        features, labels = features.to(device), labels.to(device)
        outputs, layer_outputs = model(features, return_all_layer_outputs=True)
        statistic_sum += calculate_statistic(outputs, layer_outputs, labels)
        n_examples += len(labels)

    return statistic_sum / n_examples


def calculate_average_accuracy(model, dataloader, device):
    def calculate_n_correct(outputs, layer_outputs, labels):
        class_predictions = torch.argmin(outputs, dim=-1)
        n_correct = (class_predictions == labels).sum().item()
        return n_correct
    return calculate_average_statistic(model, dataloader, device, calculate_n_correct)


def calculate_average_n_spikes(model, dataloader, device):
    def calculate_n_spikes(outputs, layer_outputs, labels):
        return sum([
            output_spike_times.isfinite().sum().item()
            for output_spike_times, _ in layer_outputs
        ])
    return calculate_average_statistic(model, dataloader, device, calculate_n_spikes)


def train_epoch(model, dataloader, criterion, optimizer, device, progress_bar=lambda x: x):
    """
    Trains the given model on the given dataloader for one epoch.
    """

    running_loss = 0.0

    model.train()
    for features, labels in progress_bar(dataloader):
        features, labels = features.to(device), labels.to(device)
        outputs = model(features)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    epoch_loss = running_loss / len(dataloader)
    return epoch_loss
