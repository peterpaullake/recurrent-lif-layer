import numpy as np
import os
from shutil import rmtree
import torch
import torchvision
from tqdm import tqdm

from recurrent_lif_layer.spike_time_solver import CUDASpikeTimeSolver
from recurrent_lif_layer.spike_time_solver import RungeKuttaSpikeTimeSolver
import training


class Experiment:
    """
    General base class for defining experiments.
    """

    def __init__(self, name):
        self.name = name
        self.base_path = os.path.join('experiments', name)

    def run(self, overwrite=False):
        """
        Runs the experiment and saves the result to the disk.
        """

        # Don't overwrite experiment data when overwrite=True
        if not overwrite and os.path.isdir(self.base_path):
            raise RuntimeError(
                f'Experiment directory "{self.base_path}" already exists. '
                'Run the experiment with overwrite=True to overwrite.'
            )

        # Completely remove the experiment directory if it exists
        try:
            rmtree(self.base_path)
        except FileNotFoundError:
            pass

        # Create a fresh experiment directory
        os.makedirs(self.base_path)

        # Run the experiment to populate the experiment directory with experiment data
        self._run()

    def _run(self):
        raise NotImplementedError

    def _save_array(self, name, array):
        """
        Saves a numpy array to the experiment directory.
        """

        np.save(os.path.join(self.base_path, name + '.npy'), array)

    def load_array(self, name):
        """
        Loads a numpy array from the experiment directory.
        """

        return np.load(os.path.join(self.base_path, name + '.npy'))


class SingleNeuronImpulseResponseExperiment(Experiment):
    """
    Probes the length of the impulse response of a neuron for different
    values of the feedforward weight w_f and recurrent weight w_r.
    """

    def __init__(self):
        super().__init__('single-neuron-impulse-response')

        w_range = 10

        wf_central = 4.0
        self.wf_min = wf_central - w_range / 2
        self.wf_max = wf_central + w_range / 2

        wr_central = 2.2
        self.wr_min = wr_central - w_range / 2
        self.wr_max = wr_central + w_range / 2

        self.grid_size = 512

    def _run(self):
        wf_values = np.linspace(self.wf_min, self.wf_max, num=self.grid_size)
        wr_values = np.linspace(self.wr_min, self.wr_max, num=self.grid_size)
        impulse_response_lengths = np.empty((self.grid_size, self.grid_size), dtype=int)

        for row_id, wr_value in enumerate(tqdm(wr_values)):
            for col_id, wf_value in enumerate(wf_values):
                device = torch.device('cuda')
                input_spike_times = torch.FloatTensor([[0.0]]).to(device)
                input_spike_synapse_ids = torch.IntTensor([[0]]).to(device)
                feedforward_weights = torch.FloatTensor([[wf_value]]).to(device)
                recurrent_weights = torch.FloatTensor([[wr_value]]).to(device)

                spike_time_solver = CUDASpikeTimeSolver()
                output_spike_times, output_spike_neuron_ids = spike_time_solver.get_output_spike_times(
                    input_spike_times,
                    input_spike_synapse_ids,
                    feedforward_weights,
                    recurrent_weights,
                    max_output_spikes=128,
                )
                impulse_response_length = output_spike_times.isfinite().sum().item()
                impulse_response_lengths[row_id, col_id] = impulse_response_length

        self._save_array('impulse-response-lengths', impulse_response_lengths)


class RegimeComparisonExperiment(Experiment):
    """
    Compares the different membrane dynamics and spiking
    behavior of a LIF layer in three different regimes:
    - single-spike without recurrence
    - multi-spike without recurrence
    - multi-spike with recurrence
    """

    def __init__(self):
        super().__init__('regime-comparison')

        self.regimes = ['single-spike', 'multi-spike', 'multi-spike with recurrence']

    def _run(self):
        max_input_spikes = 8
        n_neurons = 3
        n_synapses = 1
        max_output_spikes = 64

        np.random.seed(1)
        device = torch.device('cuda')

        input_spike_times = np.random.uniform(0, 10, size=max_input_spikes)
        n_input_spikes = np.random.randint(0, max_input_spikes + 1)
        input_spike_times[n_input_spikes:] = np.inf
        self._save_array('input-spike-times', input_spike_times)
        input_spike_times = torch.FloatTensor(input_spike_times).unsqueeze(0).to(device)
        input_spike_synapse_ids = torch.zeros((1, max_input_spikes), dtype=torch.int32).to(device)
        feedforward_weights = torch.FloatTensor(np.random.randn(n_neurons, n_synapses) + 3.3).to(device)
        recurrent_weights = torch.FloatTensor(np.random.randn(n_neurons, n_neurons) - 2).to(device)

        for regime_id, regime in enumerate(self.regimes):
            enable_multispike = regime != 'single-spike'
            enable_recurrence = regime == 'multi-spike with recurrence'

            # Calculate the output spike times
            # XXX avoiding multiple spikes using a long refractory period
            # is very hacky. Instead, we should pass the `enable_multispike`
            # flag to the solver and have it change its behavior accordingly.
            long_refractory_period = 1000.0
            if enable_multispike:
                cuda_spike_time_solver = CUDASpikeTimeSolver()
            else:
                cuda_spike_time_solver = CUDASpikeTimeSolver(refractory_period=long_refractory_period)
            output_spike_times, output_spike_neuron_ids = cuda_spike_time_solver.get_output_spike_times(
                input_spike_times,
                input_spike_synapse_ids,
                feedforward_weights,
                recurrent_weights if enable_recurrence else 0 * recurrent_weights,
                max_output_spikes,
            )

            # Convert to numpy arrays
            output_spike_times = np.array(output_spike_times[0].cpu())
            output_spike_neuron_ids = np.array(output_spike_neuron_ids[0].cpu())

            # Calculate the membrane traces
            if enable_multispike:
                rk4_spike_time_solver = RungeKuttaSpikeTimeSolver()
            else:
                rk4_spike_time_solver = RungeKuttaSpikeTimeSolver(refractory_period=long_refractory_period)
            n_output_spikes = np.isfinite(output_spike_times).sum()
            _, _, times, membrane_potentials = rk4_spike_time_solver.get_output_spike_times(
                input_spike_times,
                input_spike_synapse_ids,
                feedforward_weights,
                recurrent_weights if enable_recurrence else 0 * recurrent_weights,
                max_output_spikes=n_output_spikes,
                return_membrane_traces=True,
            )

            self._save_array(f'output-spike-times-{regime_id}', output_spike_times)
            self._save_array(f'output-spike-neuron-ids-{regime_id}', output_spike_neuron_ids)
            self._save_array(f'times-{regime_id}', times)
            self._save_array(f'membrane-potentials-{regime_id}', membrane_potentials)


class TrainingComparisonExperiment(Experiment):
    """
    Compares the training accuracy of the network in three different regimes:
    - single-spike without recurrence
    - multi-spike without recurrence
    - multi-spike with recurrence
    """

    def __init__(self):
        super().__init__('training-comparison-experiment')

        self.n_epochs = 50
        self.regimes = ['single-spike', 'multi-spike', 'multi-spike with recurrence']

    def _run(self):
        epoch_accuracies = np.empty((self.n_epochs, len(self.regimes)))
        epoch_spike_counts = np.empty((self.n_epochs, len(self.regimes)))

        for regime_id, regime in enumerate(self.regimes):
            print(f'Training regime {regime_id + 1}/{len(self.regimes)}: {regime}')

            enable_multispike = regime != 'single-spike'
            enable_recurrence = regime == 'multi-spike with recurrence'

            torch.manual_seed(0)
            device = torch.device('cuda')

            # Create a model
            model = training.SNNClassifier(
                input_size = 16 * 16,
                hidden_size=32,
                max_hidden_output_spikes=32,
                n_classes=10,
                enable_multispike=enable_multispike,
                enable_recurrence=enable_recurrence,
            )
            model.to(device)

            # Create MNIST dataloaders
            batch_size = 128

            # Use t_early and t_late from Fast and Deep
            # https://github.com/JulianGoeltz/fastAndDeep/blob/main/src/datasets.py
            t_early = 0.15
            t_late = 2.0

            # Transform the dataset: convert to tensor, downsize, flatten, affine transformation
            transform = torchvision.transforms.Compose([
                torchvision.transforms.transforms.ToTensor(),
                torchvision.transforms.Resize((16, 16), antialias=True),
                torchvision.transforms.Lambda(lambda x: x.flatten()),
                torchvision.transforms.Lambda(lambda x: t_early + (t_late - t_early) * (1.0 - x)),
            ])
            train_dataset = torchvision.datasets.MNIST(
                root='./data', train=True, transform=transform, download=True
            )
            train_dataloader = torch.utils.data.DataLoader(
                dataset=train_dataset, batch_size=batch_size, shuffle=True
            )
            validation_dataset = torchvision.datasets.MNIST(
                root='./data', train=False, transform=transform, download=True
            )
            validation_dataloader = torch.utils.data.DataLoader(
                dataset=validation_dataset, batch_size=batch_size, shuffle=False
            )

            # Create a criterion and optimizer
            criterion = lambda outputs, labels: training.TTFSLoss.apply(outputs, labels).mean()
            optimizer = torch.optim.Adam(model.parameters())

            # Train
            for epoch_id in range(self.n_epochs):
                print(f'Training epoch {epoch_id + 1}/{self.n_epochs}')
                training.train_epoch(model, train_dataloader, criterion, optimizer, device, tqdm)
                epoch_accuracies[epoch_id, regime_id] = training.calculate_average_accuracy(
                    model, validation_dataloader, device
                )
                epoch_spike_counts[epoch_id, regime_id] = training.calculate_average_n_spikes(
                    model, validation_dataloader, device
                )

        self._save_array('accuracies', epoch_accuracies)
        self._save_array('spike-counts', epoch_spike_counts)
