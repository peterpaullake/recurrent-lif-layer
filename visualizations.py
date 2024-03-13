import matplotlib.pyplot as plt
import numpy as np
import os
from shutil import rmtree

import experiments


class Visualization:
    def __init__(self, name, latex_target='report'):
        assert latex_target in ['report', 'slides']

        self.name = name
        self.latex_target = latex_target
        self.base_path = os.path.join('visualizations', name, latex_target)

        # Use the matplotlib settings from Jack Walton
        # https://jwalton.info/Embed-Publication-Matplotlib-Latex/
        tex_fonts = {
            # Use LaTeX to write all text
            'text.usetex': True,
            'font.family': 'serif',
            # Include amsmath so we use the \text command
            # 'text.latex.preamble': r'\usepackage{amsmath}',
            # Use 10pt font in plots to match 10pt font in document
            'axes.labelsize': 10,
            'font.size': 10,
            # Make the legend/label fonts a little smaller
            'legend.fontsize': 8,
            'xtick.labelsize': 8,
            'ytick.labelsize': 8,
        }
        plt.rcParams.update(tex_fonts)

    def create(self, dpi=300):
        """
        Creates the visualization and saves it to the disk.
        """

        # Completely remove the output directory if it exists
        try:
            rmtree(self.base_path)
        except FileNotFoundError:
            pass

        # Create a fresh output directory
        os.makedirs(self.base_path)

        # Run the visualization code
        result = self._create()

        # Save the matplotlib figure or figures to the output directory
        if isinstance(result, plt.Figure):
            result.savefig(os.path.join(self.base_path, self.name + '.pgf'), dpi=dpi)
            plt.close(result)
        elif isinstance(result, list) and all([isinstance(x, plt.Figure) for x in result]):
            for i, x in enumerate(result):
                x.savefig(os.path.join(self.base_path, self.name + f'-{i + 1}.pgf'), dpi=dpi)
                plt.close(x)

        return result

    def _create(self):
        raise NotImplementedError

    def calculate_figsize(self, page_width_fraction, height_to_width=None, n_rows=1, n_cols=1):
        """
        Calculates the correct `figsize` argument for the
        `plt.subplots` function. Adapted from Jack Walton.
        https://jwalton.info/Embed-Publication-Matplotlib-Latex/
        
        page_width_fraction : fraction of the \textwidth that the figure should occupy
        height_to_width     : height-to-width ratio of each subplot
        n_rows              : number of subplot rows
        n_cols              : number of subplot cols
        """

        # Use the golden ratio for the subplot dimensions by default
        if height_to_width is None:
            height_to_width = 2 / (1 + np.sqrt(5))

        page_width_pt = {
            'report': 452.9679, # \textwidth of an A4 LaTeX document with 1 inch margins
            'slides': 352.81429, # \textwidth of a Beamer slide
        }[self.latex_target]

        # Calculate the figure dimensions in points
        figure_width_pt = page_width_fraction * page_width_pt
        figure_height_pt = figure_width_pt / n_cols * height_to_width * n_rows

        # Calculate the figure dimensions in inches
        inches_per_pt = 1 / 72.27
        figure_width_in = figure_width_pt * inches_per_pt
        figure_height_in = figure_height_pt * inches_per_pt

        return figure_width_in, figure_height_in


class SingleNeuronImpulseResponseVisualization(Visualization):
    def __init__(self):
        super().__init__('single-neuron-impulse-response')

    def _create(self):
        experiment = experiments.SingleNeuronImpulseResponseExperiment()
        fig, ax = plt.subplots(figsize=self.calculate_figsize(
            page_width_fraction=0.9,
            height_to_width=1.0,
            n_rows=1, n_cols=1,
        ))

        ax.set_xlabel(r'$w^f$')
        ax.set_ylabel(r'$w^r$')

        grid_size = 64
        xs, ys = np.meshgrid(
            np.linspace(0, 10 * 2 * np.pi, grid_size),
            np.linspace(0, 10 * 2 * np.pi, grid_size),
        )

        # Plot the impulse response lengths
        impulse_response_lengths = experiment.load_array('impulse-response-lengths')
        im = ax.imshow(
            np.log2(impulse_response_lengths + 1),
            extent=(experiment.wf_min, experiment.wf_max, experiment.wr_min, experiment.wr_max),
            origin='lower',
            cmap='magma'
        )

        # Add a colorbar
        # https://matplotlib.org/stable/gallery/axes_grid1/simple_colorbar.html
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        colorbar = plt.colorbar(
            im,
            cax=make_axes_locatable(ax).append_axes(
                'right', size='5%', pad=0.05
            ),
        )
        colorbar.ax.set_ylabel(r'Number of emitted spikes')

        # Make the have the correct scale
        # https://matplotlib.org/3.4.3/gallery/ticks_and_spines/colorbar_tick_labelling_demo.html
        colorbar.ax.set_yticks(
            np.linspace(
                np.log2(impulse_response_lengths + 1).min(),
                np.log2(impulse_response_lengths + 1).max(),
                num=10,
            )
        )
        colorbar.ax.set_yticklabels(
            [f'{(2**x - 1):.2f}' for x in colorbar.ax.get_yticks()]
        )

        fig.tight_layout()
        return fig

class RegimeComparisonVisualization(Visualization):
    def __init__(self):
        super().__init__('regime-comparison')

    def _create(self):
        experiment = experiments.RegimeComparisonExperiment()
        fig, axes = plt.subplots(
            len(experiment.regimes),
            sharex=True,
            figsize=self.calculate_figsize(
                page_width_fraction=1.0,
                height_to_width=1/3,
                n_rows=len(experiment.regimes), n_cols=1,
            )
        )

        for regime_id, (regime, ax) in enumerate(zip(experiment.regimes, axes)):
            output_spike_times = experiment.load_array(f'output-spike-times-{regime_id}')
            output_spike_neuron_ids = experiment.load_array(f'output-spike-neuron-ids-{regime_id}')
            times = experiment.load_array(f'times-{regime_id}')
            membrane_potentials = experiment.load_array(f'membrane-potentials-{regime_id}')

            # Colors taken from Figure 1 of Fast and Deep paper
            neuron_colors = ['#1f77b4', '#d62728', '#2ca02c']
            for neuron_id, neuron_color in enumerate(neuron_colors):
                # Plot the membrane potential
                ax.plot(times, membrane_potentials[:, neuron_id], alpha=0.8, color=neuron_color)

                # Plot the input spike times
                input_spike_times = experiment.load_array('input-spike-times')
                for input_spike_time in input_spike_times:
                    ax.axvline(input_spike_time, alpha=0.2, color='grey', linestyle='--')

                # Plot the output spike times
                neuron_spike_times = output_spike_times[np.logical_and(
                    np.isfinite(output_spike_times),
                    output_spike_neuron_ids == neuron_id
                )]
                for neuron_spike_time in neuron_spike_times:
                    ax.plot([neuron_spike_time], [1.0], '.', color=neuron_color)

        # Add decorations
        regime_titles = {
            'single-spike': 'a) Single-spike without recurrence',
            'multi-spike': 'b) Multi-spike without recurrence',
            'multi-spike with recurrence': 'c) Multi-spike with recurrence',
        }
        for ax, regime in zip(axes, experiment.regimes):
            ax.set_title(regime_titles[regime])

        fig.tight_layout()
        return fig


class TrainingComparisonVisualization(Visualization):
    def __init__(self):
        super().__init__('training-comparison')

    def _create(self):
        experiment = experiments.TrainingComparisonExperiment()
        arrays = {
            'accuracies': experiment.load_array('accuracies'),
            'spike-counts': experiment.load_array('spike-counts'),
        }
        fig, axes = plt.subplots(
            2,
            len(experiment.regimes),
            figsize=self.calculate_figsize(
                page_width_fraction=1.0,
                height_to_width=0.9,
                n_rows=2,
                n_cols=len(experiment.regimes),
        ))

        # Manually share y-axes for each row
        for ax_row in axes:
            for i in range(len(ax_row) - 1):
                ax_row[i].sharey(ax_row[i + 1])

        # Plot the arrays
        for array_id, (array_name, ax_row) in enumerate(zip(['accuracies', 'spike-counts'], axes)):
            array = arrays[array_name]
            for regime_id, (regime, ax) in enumerate(zip(experiment.regimes, ax_row)):
                n_epochs = len(array)
                epochs = np.arange(n_epochs)
                ax.plot(epochs, array[:, regime_id])

        # Add decorations
        regime_titles = {
            'single-spike': 'a) Single-spike\nwithout recurrence',
            'multi-spike': 'b) Multi-spike\nwithout recurrence',
            'multi-spike with recurrence': 'c) Multi-spike\nwith recurrence',
        }
        for ax, regime in zip(axes[0], experiment.regimes):
            ax.set_title(regime_titles[regime])
        axes[0, 0].set_ylabel('Validation accuracy')
        axes[1, 0].set_ylabel('Validation spike count\nper classification')
        for ax in axes[-1]:
            ax.set_xlabel('Epochs')

        fig.tight_layout()
        return fig
