#include <torch/extension.h>

std::vector<torch::Tensor> get_output_spike_times(
    torch::Tensor input_spike_times,
    torch::Tensor input_spike_synapse_ids,
    torch::Tensor feedforward_weights,
    torch::Tensor recurrent_weights,
    int max_output_spikes,
    float leak_conductance,
    float tau_membrane,
    float tau_synapse,
    float threshold_potential,
    float reset_potential,
    float refractory_period
);

torch::Tensor get_input_spike_time_derivatives(
    torch::Tensor input_spike_times,
    torch::Tensor input_spike_synapse_ids,
    torch::Tensor output_spike_times,
    torch::Tensor output_spike_neuron_ids,
    torch::Tensor feedforward_weights,
    torch::Tensor recurrent_weights,
    float leak_conductance,
    float tau_membrane,
    float tau_synapse,
    float threshold_potential,
    float reset_potential,
    float refractory_period
);

torch::Tensor get_feedforward_weight_derivatives(
    torch::Tensor input_spike_times,
    torch::Tensor input_spike_synapse_ids,
    torch::Tensor output_spike_times,
    torch::Tensor output_spike_neuron_ids,
    torch::Tensor feedforward_weights,
    torch::Tensor recurrent_weights,
    float leak_conductance,
    float tau_membrane,
    float tau_synapse,
    float threshold_potential,
    float reset_potential,
    float refractory_period
);

torch::Tensor get_recurrent_weight_derivatives(
    torch::Tensor input_spike_times,
    torch::Tensor input_spike_synapse_ids,
    torch::Tensor output_spike_times,
    torch::Tensor output_spike_neuron_ids,
    torch::Tensor feedforward_weights,
    torch::Tensor recurrent_weights,
    float leak_conductance,
    float tau_membrane,
    float tau_synapse,
    float threshold_potential,
    float reset_potential,
    float refractory_period
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "get_output_spike_times", &get_output_spike_times,
	"See recurrent_lif_layer.spike_time_solver.CUDASpikeTimeSolver.get_output_spike_times for example usage."
    );
    m.def(
        "get_input_spike_time_derivatives", &get_input_spike_time_derivatives,
	"See recurrent_lif_layer.derivative_evaluator.CUDADerivativeEvaluator.get_input_spike_time_derivatives for example usage."
    );
    m.def(
        "get_feedforward_weight_derivatives", &get_feedforward_weight_derivatives,
	"See recurrent_lif_layer.derivative_evaluator.CUDADerivativeEvaluator.get_feedforward_weight_derivatives for example usage."
    );
    m.def(
        "get_recurrent_weight_derivatives", &get_recurrent_weight_derivatives,
	"See recurrent_lif_layer.derivative_evaluator.CUDADerivativeEvaluator.get_recurrent_weight_derivatives for example usage."
    );
}
