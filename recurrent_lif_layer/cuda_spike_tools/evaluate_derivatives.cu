#include <torch/extension.h>

#define EPSILON 1e-6
#define MAX_NEURONS 256

// The ampersand `&` indicates that any variables from
// the surrounding scope used in the lambda function
// should be captured by reference as opposed to by value
#define SAFE_DIVIDE(numerator, denominator) \
    ([&]() { \
        float _evaluated_denominator = (denominator); \
        return (fabs(_evaluated_denominator) < EPSILON) ? 0.0f : (numerator) / _evaluated_denominator; \
    }())

class DerivativeEvaluator {
private:
    const float *m_input_spike_times;
    const int *m_input_spike_synapse_ids;
    const int *m_input_spike_sorted_ids;
    const float *m_output_spike_times;
    const int *m_output_spike_neuron_ids;
    const float *m_feedforward_weights;
    const float *m_recurrent_weights;
    int m_max_input_spikes;
    int m_max_output_spikes;
    int m_n_neurons;
    int m_n_synapses;
    float m_leak_conductance;
    float m_tau_membrane;
    float m_tau_synapse;
    float m_threshold_potential;
    float m_reset_potential;
    float m_refractory_period;
    float m_prefactor;
    float m_epsilon;

    __device__ float get_feedforward_weight(int neuron_id, int synapse_id) const {
	return m_feedforward_weights[neuron_id * m_n_synapses + synapse_id];
    }

    __device__ float get_recurrent_weight(int target_neuron_id, int source_neuron_id) const {
	return m_recurrent_weights[target_neuron_id * m_n_neurons + source_neuron_id];
    }

    __device__ float evaluate_weighted_kernel_sum(
        float (DerivativeEvaluator::*kernel)(float) const, float t, int neuron_id
    ) const {
	return evaluate_weighted_kernel_sum(
            kernel, t, neuron_id,
	    /*keep_synapse_id=*/-1,
	    /*keep_neuron_id=*/-1,
	    /*use_weight=*/true,
	    /*additional_recurrent_weights=*/nullptr
        );
    }

    __device__ float evaluate_kernel_sum(
        float (DerivativeEvaluator::*kernel)(float) const,
	float t, int neuron_id,
	int keep_synapse_id,
	int keep_neuron_id
    ) const {
	return evaluate_weighted_kernel_sum(
            kernel, t, neuron_id,
	    /*keep_synapse_id=*/keep_synapse_id,
	    /*keep_neuron_id=*/keep_neuron_id,
	    /*use_weight=*/false,
	    /*additional_recurrent_weights=*/nullptr
        );
    }

    __device__ float evaluate_weighted_kernel_sum(
        float (DerivativeEvaluator::*kernel)(float) const,
	float t, int neuron_id,
	const float* additional_recurrent_weights
    ) const {
	return evaluate_weighted_kernel_sum(
            kernel, t, neuron_id,
	    /*keep_synapse_id=*/-1,
	    /*keep_neuron_id=*/-1,
	    /*use_weight=*/true,
	    /*additional_recurrent_weights=*/additional_recurrent_weights
        );
    }
	
    __device__ float evaluate_weighted_kernel_sum(
        float (DerivativeEvaluator::*kernel)(float) const,
	float t, int neuron_id,
	int keep_synapse_id,
	int keep_neuron_id,
	bool use_weight,
	const float* additional_recurrent_weights
    ) const {
	float sum = 0.0f;

	bool skip_feedforward_terms = keep_neuron_id != -1 || additional_recurrent_weights != nullptr;
	if (!skip_feedforward_terms) {
	    // Include the feedforward terms, i.e. the terms due to the input spikes
	    for (int input_spike_id = 0; input_spike_id < m_max_input_spikes; input_spike_id++) {
		int spike_synapse_id = m_input_spike_synapse_ids[m_input_spike_sorted_ids[input_spike_id]];
		if (keep_synapse_id != -1 && spike_synapse_id != keep_synapse_id) {
		    continue;
		}

		float input_spike_time = m_input_spike_times[m_input_spike_sorted_ids[input_spike_id]];
		if (isinf(input_spike_time) || input_spike_time >= t) {
		    break;
		}

		float weight;
		if (use_weight) {
		    weight = get_feedforward_weight(neuron_id, spike_synapse_id);
		} else {
		    weight = 1.0f;
		}
		sum += weight * (this->*kernel)(t - input_spike_time);
	    }
	}

	bool skip_recurrent_terms = keep_synapse_id != -1;
	if (!skip_recurrent_terms) {
	    // Include the recurrent terms, i.e. the terms due to the output spikes
	    for (int output_spike_id = 0; output_spike_id < m_max_output_spikes; output_spike_id++) {
		int spike_neuron_id = m_output_spike_neuron_ids[output_spike_id];
		if (keep_neuron_id != -1 && spike_neuron_id != keep_neuron_id) {
		    continue;
		}

		float output_spike_time = m_output_spike_times[output_spike_id];
		if (isinf(output_spike_time) || output_spike_time >= t) {
		    break;
		}

		float weight;
		if (use_weight) {
		    weight = get_recurrent_weight(neuron_id, spike_neuron_id);
		} else {
		    weight = 1.0f;
		}
		if (additional_recurrent_weights != nullptr) {
		    weight *= additional_recurrent_weights[output_spike_id];
		}
		sum += weight * (this->*kernel)(t - output_spike_time);
	    }
	}

	return sum;
    }

    __device__ float kappa(float t) const {
	if (t <= 0.0f) {
	    return 0.0f;
	} else {
	    return expf(-t / m_tau_membrane) - expf(-t / m_tau_synapse);
	}
    }

    __device__ float kappa_derivative(float t) const {
	if (t <= 0.0f) {
	    return 0.0f;
	} else {
	    return -expf(-t / m_tau_membrane) / m_tau_membrane + expf(-t / m_tau_synapse) / m_tau_synapse;
	}
    }

    __device__ float evaluate_initial_condition_factor_derivative_with_respect_to_input_spike_time(
	int output_spike_id,
	int input_spike_id,
	float prev_output_spike_time,
	float initial_condition_factor,
	float prev_input_spike_time_derivative,
	const float *input_spike_time_derivatives
    ) const {
	float input_spike_time = m_input_spike_times[input_spike_id];
	float t_initial = prev_output_spike_time + m_refractory_period;
	if (isnan(prev_output_spike_time) || input_spike_time >= t_initial) {
	    return 0.0f;
	}

	int output_spike_neuron_id = m_output_spike_neuron_ids[output_spike_id];
	float derivative = 0.0f;
	derivative += prev_input_spike_time_derivative * (
            initial_condition_factor / m_tau_membrane
	    - m_prefactor * evaluate_weighted_kernel_sum(
                  &DerivativeEvaluator::kappa_derivative, t_initial, output_spike_neuron_id
              ) * expf(t_initial / m_tau_membrane)
        );

	int input_spike_synapse_id = m_input_spike_synapse_ids[input_spike_id];
	float weight = get_feedforward_weight(output_spike_neuron_id, input_spike_synapse_id);
	derivative += (
            m_prefactor * weight * kappa_derivative(t_initial - input_spike_time)
	    + evaluate_weighted_kernel_sum(
                &DerivativeEvaluator::kappa_derivative,
	        t_initial, output_spike_neuron_id,
	        /*additional_recurrent_weights=*/input_spike_time_derivatives
            )
	) * expf(t_initial / m_tau_membrane);

	return derivative;
    }

    __device__ float evaluate_input_spike_time_derivative(
	int output_spike_id,
	int input_spike_id,
	float prev_output_spike_time,
	float initial_condition_factor,
	float prev_input_spike_time_derivative,
	const float *input_spike_time_derivatives
    ) const {
	/*
	 * XXX you should clearly note here and wherever else uses it that `prev_output_spike_time` is
	 * the previous output spike time of the neuron being considered or NaN if no such spike exists.
	 */

	float output_spike_time = m_output_spike_times[output_spike_id];
	int output_spike_neuron_id = m_output_spike_neuron_ids[output_spike_id];
	float input_spike_time = m_input_spike_times[input_spike_id];
	int input_spike_synapse_id = m_input_spike_synapse_ids[input_spike_id];

	if (isinf(output_spike_time) || isinf(input_spike_time) || input_spike_time >= output_spike_time) {
	    return 0.0f;
	}

	float weight = get_feedforward_weight(output_spike_neuron_id, input_spike_synapse_id);
	float dI_dt = evaluate_initial_condition_factor_derivative_with_respect_to_input_spike_time(
            output_spike_id,
	    input_spike_id,
	    prev_output_spike_time,
	    initial_condition_factor,
	    prev_input_spike_time_derivative,
	    input_spike_time_derivatives
        );
	float feedforward_term = evaluate_weighted_kernel_sum(
            &DerivativeEvaluator::kappa_derivative,
	    output_spike_time, output_spike_neuron_id,
	    /*additional_recurrent_weights=*/input_spike_time_derivatives
        );
	float numerator = (
            weight * kappa_derivative(output_spike_time - input_spike_time)
	    - dI_dt / m_prefactor * expf(-output_spike_time / m_tau_membrane)
	    + feedforward_term
        );
	float denominator = (
            evaluate_weighted_kernel_sum(
                &DerivativeEvaluator::kappa_derivative, output_spike_time, output_spike_neuron_id
            )
	    - (
		1.0f / m_tau_membrane
                * initial_condition_factor / m_prefactor
	        * expf(-output_spike_time / m_tau_membrane)
	    )
        );

	return SAFE_DIVIDE(numerator, denominator);
    }

    __device__ float evaluate_initial_condition_factor_derivative_with_respect_to_feedforward_weight(
        int output_spike_id,
	int neuron_id, int synapse_id,
	float prev_output_spike_time,
	float initial_condition_factor,
	float prev_feedforward_weight_derivative,
	const float *feedforward_weight_derivatives
    ) const {
	float t_initial = prev_output_spike_time + m_refractory_period;
	if (isnan(prev_output_spike_time)) {
	    return 0.0f;
	}

	int output_spike_neuron_id = m_output_spike_neuron_ids[output_spike_id];
	float dI_dw = 0.0f;

	dI_dw -= prev_feedforward_weight_derivative * m_prefactor * evaluate_weighted_kernel_sum(
            &DerivativeEvaluator::kappa_derivative, t_initial, output_spike_neuron_id
        ) * expf(t_initial / m_tau_membrane);

	dI_dw += m_prefactor * evaluate_weighted_kernel_sum(
            &DerivativeEvaluator::kappa_derivative, t_initial, output_spike_neuron_id,
	    /*additional_recurrent_weights=*/feedforward_weight_derivatives
        ) * expf(t_initial / m_tau_membrane);

	dI_dw += initial_condition_factor / m_tau_membrane * prev_feedforward_weight_derivative;

	if (output_spike_neuron_id == neuron_id) {
	    dI_dw -= evaluate_kernel_sum(
                &DerivativeEvaluator::kappa,
	        t_initial, output_spike_neuron_id,
	        /*keep_synapse_id=*/synapse_id,
		/*keep_neuron_id=*/-1
            ) * expf(t_initial / m_tau_membrane);
	}

	return dI_dw;
    }

    __device__ float evaluate_feedforward_weight_derivative(
        int output_spike_id,
	int neuron_id, int synapse_id,
	float prev_output_spike_time,
	float initial_condition_factor,
	float prev_feedforward_weight_derivative,
	const float *feedforward_weight_derivatives
    ) const {
	float output_spike_time = m_output_spike_times[output_spike_id];
	int output_spike_neuron_id = m_output_spike_neuron_ids[output_spike_id];

	float numerator = 0.0f;

	// Include the indirect influence from the preceding output spikes
	numerator += evaluate_weighted_kernel_sum(
            &DerivativeEvaluator::kappa_derivative, output_spike_time, output_spike_neuron_id,
	    /*additional_recurrent_weights=*/feedforward_weight_derivatives
        );

	// Include the influence from the initial condition
	float dI_dw = evaluate_initial_condition_factor_derivative_with_respect_to_feedforward_weight(
            output_spike_id,
	    neuron_id,
	    synapse_id,
	    prev_output_spike_time,
	    initial_condition_factor,
	    prev_feedforward_weight_derivative,
	    feedforward_weight_derivatives
        );
	numerator -= dI_dw / m_prefactor * expf(-output_spike_time / m_tau_membrane);

	// Include the direct influence from the preceding input spikes
	if (output_spike_neuron_id == neuron_id) {
	    numerator -= evaluate_kernel_sum(
                &DerivativeEvaluator::kappa,
	        output_spike_time, output_spike_neuron_id,
	        /*keep_synapse_id=*/synapse_id,
		/*keep_neuron_id=*/-1
            );
	}

	float denominator = 0.0f;

	denominator += evaluate_weighted_kernel_sum(
            &DerivativeEvaluator::kappa_derivative,
	    output_spike_time, output_spike_neuron_id
        );

	denominator -= (
            1.0f / m_tau_membrane
            * initial_condition_factor / m_prefactor
	    * expf(-output_spike_time / m_tau_membrane)
        );

	return SAFE_DIVIDE(numerator, denominator);
    }

    __device__ float evaluate_initial_condition_factor_derivative_with_respect_to_recurrent_weight(
        int output_spike_id,
        int target_neuron_id,
	int source_neuron_id,
	float prev_output_spike_time,
	float initial_condition_factor,
	float prev_recurrent_weight_derivative,
	const float *recurrent_weight_derivatives
    ) const {
	float t_initial = prev_output_spike_time + m_refractory_period;
	if (isnan(prev_output_spike_time)) {
	    return 0.0f;
	}

	int output_spike_neuron_id = m_output_spike_neuron_ids[output_spike_id];
	float dI_dv = 0.0f;

	dI_dv -= prev_recurrent_weight_derivative * m_prefactor * evaluate_weighted_kernel_sum(
            &DerivativeEvaluator::kappa_derivative, t_initial, output_spike_neuron_id
        ) * expf(t_initial / m_tau_membrane);

	dI_dv += m_prefactor * evaluate_weighted_kernel_sum(
            &DerivativeEvaluator::kappa_derivative, t_initial, output_spike_neuron_id,
	    /*additional_recurrent_weights=*/recurrent_weight_derivatives
        ) * expf(t_initial / m_tau_membrane);

	dI_dv += initial_condition_factor / m_tau_membrane * prev_recurrent_weight_derivative;

	if (output_spike_neuron_id == target_neuron_id) {
	    dI_dv -= evaluate_kernel_sum(
                &DerivativeEvaluator::kappa,
	        t_initial, output_spike_neuron_id,
	        /*keep_synapse_id=*/-1,
		/*keep_neuron_id=*/source_neuron_id
            ) * expf(t_initial / m_tau_membrane);
	}

	return dI_dv;
    }

    __device__ float evaluate_recurrent_weight_derivative(
        int output_spike_id,
        int target_neuron_id,
	int source_neuron_id,
	float prev_output_spike_time,
	float initial_condition_factor,
	float prev_recurrent_weight_derivative,
	const float *recurrent_weight_derivatives
    ) const {
	float output_spike_time = m_output_spike_times[output_spike_id];
	int output_spike_neuron_id = m_output_spike_neuron_ids[output_spike_id];

	float numerator = 0.0f;

	// Include the indirect influence from the preceding output spikes
	numerator += evaluate_weighted_kernel_sum(
            &DerivativeEvaluator::kappa_derivative, output_spike_time, output_spike_neuron_id,
	    /*additional_recurrent_weights=*/recurrent_weight_derivatives
        );

	// Include the influence from the initial condition
	float dI_dv = evaluate_initial_condition_factor_derivative_with_respect_to_recurrent_weight(
            output_spike_id,
	    target_neuron_id,
	    source_neuron_id,
	    prev_output_spike_time,
	    initial_condition_factor,
	    prev_recurrent_weight_derivative,
	    recurrent_weight_derivatives
        );
	numerator -= dI_dv / m_prefactor * expf(-output_spike_time / m_tau_membrane);

	// Include the direct influence from the preceding output spikes
	if (output_spike_neuron_id == target_neuron_id) {
	    numerator -= evaluate_kernel_sum(
                &DerivativeEvaluator::kappa,
	        output_spike_time, output_spike_neuron_id,
	        /*keep_synapse_id=*/-1,
		/*keep_neuron_id=*/source_neuron_id
            );
	}

	float denominator = 0.0f;

	denominator += evaluate_weighted_kernel_sum(
            &DerivativeEvaluator::kappa_derivative,
	    output_spike_time, output_spike_neuron_id
        );

	denominator -= (
            1.0f / m_tau_membrane
            * initial_condition_factor / m_prefactor
	    * expf(-output_spike_time / m_tau_membrane)
        );

	return SAFE_DIVIDE(numerator, denominator);
    }

    __device__ float evaluate_initial_condition_factor(float t0, float u0, int neuron_id) const {
	return (u0 - m_prefactor * evaluate_weighted_kernel_sum(
            &DerivativeEvaluator::kappa, t0, neuron_id
        )) * expf(t0 / m_tau_membrane);
    }

public:
    __device__ DerivativeEvaluator (
        const float *input_spike_times,
        const int *input_spike_synapse_ids,
        const int *input_spike_sorted_ids,
        const float *output_spike_times,
        const int *output_spike_neuron_ids,
        const float *feedforward_weights,
        const float *recurrent_weights,
        int max_input_spikes,
        int max_output_spikes,
	int n_neurons,
	int n_synapses,
        float leak_conductance,
        float tau_membrane,
        float tau_synapse,
        float threshold_potential,
        float reset_potential,
        float refractory_period,
        float epsilon
    )
	: m_input_spike_times(input_spike_times),
	  m_input_spike_synapse_ids(input_spike_synapse_ids),
	  m_input_spike_sorted_ids(input_spike_sorted_ids),
	  m_output_spike_times(output_spike_times),
	  m_output_spike_neuron_ids(output_spike_neuron_ids),
	  m_feedforward_weights(feedforward_weights),
	  m_recurrent_weights(recurrent_weights),
	  m_max_input_spikes(max_input_spikes),
          m_max_output_spikes(max_output_spikes),
	  m_n_neurons(n_neurons),
	  m_n_synapses(n_synapses),
	  m_leak_conductance(leak_conductance),
	  m_tau_membrane(tau_membrane),
	  m_tau_synapse(tau_synapse),
	  m_threshold_potential(threshold_potential),
	  m_reset_potential(reset_potential),
	  m_refractory_period(refractory_period),
	  m_epsilon(epsilon) {
	float membrane_capacitance = m_leak_conductance * m_tau_membrane;
	m_prefactor = (
            1.0f
	    / membrane_capacitance
	    * m_tau_membrane * m_tau_synapse
	    / (m_tau_membrane - m_tau_synapse)
        );
    }

    __device__ void evaluate_input_spike_time_derivatives(
        int input_spike_id, float *input_spike_time_derivatives
    ) const {
	float input_spike_time = m_input_spike_times[input_spike_id];
	float prev_output_spike_times[MAX_NEURONS];
	float initial_condition_factors[MAX_NEURONS] = {0.0f};
	float prev_derivatives[MAX_NEURONS] = {0.0f};

	assert(m_n_neurons <= MAX_NEURONS);

	for (int neuron_id = 0; neuron_id < m_n_neurons; neuron_id++) {
	    prev_output_spike_times[neuron_id] = NAN;
	}

	for (int output_spike_id = 0; output_spike_id < m_max_output_spikes; output_spike_id++) {
	    float output_spike_time = m_output_spike_times[output_spike_id];
	    int output_spike_neuron_id = m_output_spike_neuron_ids[output_spike_id];
	    float derivative = 0.0f;

	    if (!isinf(output_spike_time) && !isinf(input_spike_time)) {
		float prev_output_spike_time = prev_output_spike_times[output_spike_neuron_id];
		float initial_condition_factor = initial_condition_factors[output_spike_neuron_id];

		derivative = evaluate_input_spike_time_derivative(
                    output_spike_id,
		    input_spike_id,
		    prev_output_spike_time,
		    initial_condition_factor,
		    prev_derivatives[output_spike_neuron_id],
		    input_spike_time_derivatives
                );

		// Save the output spike time for future use
		prev_output_spike_times[output_spike_neuron_id] = output_spike_time;

		// Calculate the new initial condition factor based on the output spike we just processed
		initial_condition_factors[output_spike_neuron_id] = evaluate_initial_condition_factor(
                    output_spike_time + m_refractory_period, m_reset_potential, output_spike_neuron_id
                );

		// Save the new derivative for future use
		prev_derivatives[output_spike_neuron_id] = derivative;
	    }

	    input_spike_time_derivatives[output_spike_id] = derivative;
	}
    }

    __device__ void evaluate_feedforward_weight_derivatives(
        int neuron_id, int synapse_id, float *feedforward_weight_derivatives
    ) const {
	float prev_output_spike_times[MAX_NEURONS];
	float initial_condition_factors[MAX_NEURONS] = {0.0f};
	float prev_derivatives[MAX_NEURONS] = {0.0f};

	assert(m_n_neurons <= MAX_NEURONS);

	for (int neuron_id = 0; neuron_id < m_n_neurons; neuron_id++) {
	    prev_output_spike_times[neuron_id] = NAN;
	}

	for (int output_spike_id = 0; output_spike_id < m_max_output_spikes; output_spike_id++) {
	    float output_spike_time = m_output_spike_times[output_spike_id];
	    int output_spike_neuron_id = m_output_spike_neuron_ids[output_spike_id];
	    float derivative = 0.0f;

	    if (!isinf(output_spike_time)) {
		float prev_output_spike_time = prev_output_spike_times[output_spike_neuron_id];
		float initial_condition_factor = initial_condition_factors[output_spike_neuron_id];

		derivative = evaluate_feedforward_weight_derivative(
                    output_spike_id,
		    neuron_id, synapse_id,
		    prev_output_spike_time,
		    initial_condition_factor,
		    prev_derivatives[output_spike_neuron_id],
		    feedforward_weight_derivatives
		);

		// Save the output spike time for future use
		prev_output_spike_times[output_spike_neuron_id] = output_spike_time;

		// Calculate the new initial condition factor based on the output spike we just processed
		initial_condition_factors[output_spike_neuron_id] = evaluate_initial_condition_factor(
                    output_spike_time + m_refractory_period, m_reset_potential, output_spike_neuron_id
                );

		// Save the new derivative for future use
		prev_derivatives[output_spike_neuron_id] = derivative;
	    }

	    feedforward_weight_derivatives[output_spike_id] = derivative;
	}
    }

    __device__ void evaluate_recurrent_weight_derivatives(
        int target_neuron_id, int source_neuron_id, float *recurrent_weight_derivatives
    ) const {
	float prev_output_spike_times[MAX_NEURONS];
	float initial_condition_factors[MAX_NEURONS] = {0.0f};
	float prev_derivatives[MAX_NEURONS] = {0.0f};

	assert(m_n_neurons <= MAX_NEURONS);

	for (int neuron_id = 0; neuron_id < m_n_neurons; neuron_id++) {
	    prev_output_spike_times[neuron_id] = NAN;
	}

	for (int output_spike_id = 0; output_spike_id < m_max_output_spikes; output_spike_id++) {
	    float output_spike_time = m_output_spike_times[output_spike_id];
	    int output_spike_neuron_id = m_output_spike_neuron_ids[output_spike_id];
	    float derivative = 0.0f;

	    if (!isinf(output_spike_time)) {
		float prev_output_spike_time = prev_output_spike_times[output_spike_neuron_id];
		float initial_condition_factor = initial_condition_factors[output_spike_neuron_id];

		derivative = evaluate_recurrent_weight_derivative(
		    output_spike_id,
		    target_neuron_id,
		    source_neuron_id,
		    prev_output_spike_time,
		    initial_condition_factor,
		    prev_derivatives[output_spike_neuron_id],
		    recurrent_weight_derivatives
		);

		// Save the output spike time for future use
		prev_output_spike_times[output_spike_neuron_id] = output_spike_time;

		// Calculate the new initial condition factor based on the output spike we just processed
		initial_condition_factors[output_spike_neuron_id] = evaluate_initial_condition_factor(
                    output_spike_time + m_refractory_period, m_reset_potential, output_spike_neuron_id
                );

		// Save the new derivative for future use
		prev_derivatives[output_spike_neuron_id] = derivative;
	    }

	    recurrent_weight_derivatives[output_spike_id] = derivative;
	}
    }
};

__global__ void get_input_spike_time_derivatives_kernel(
    // Input and output arrays
    const float *input_spike_times,
    const int *input_spike_synapse_ids,
    const int *input_spike_sorted_ids,
    const float *output_spike_times,
    const int *output_spike_neuron_ids,
    const float *feedforward_weights,
    const float *recurrent_weights,
    float *input_spike_time_derivatives,
    // Array dimensions
    int batch_size,
    int max_input_spikes,
    int max_output_spikes,
    int n_neurons,
    int n_synapses,
    // Neuron parameters
    float leak_conductance,
    float tau_membrane,
    float tau_synapse,
    float threshold_potential,
    float reset_potential,
    float refractory_period
) {
    const int global_thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    const int example_id = global_thread_id / max_input_spikes;
    const int input_spike_id = global_thread_id % max_input_spikes;
    const float epsilon = 1e-6f;

    // Return if this outside the bounds of the problem
    if (example_id >= batch_size || input_spike_id >= max_input_spikes) {
	return;
    }

    // Look at the parts of the arrays corresponding to this particular example and input spike
    input_spike_times = &input_spike_times[example_id * max_input_spikes];
    input_spike_synapse_ids = &input_spike_synapse_ids[example_id * max_input_spikes];
    input_spike_sorted_ids = &input_spike_sorted_ids[example_id * max_input_spikes];
    output_spike_times = &output_spike_times[example_id * max_output_spikes];
    output_spike_neuron_ids = &output_spike_neuron_ids[example_id * max_output_spikes];
    input_spike_time_derivatives = &input_spike_time_derivatives[global_thread_id * max_output_spikes];

    const DerivativeEvaluator derivative_evaluator(
        input_spike_times,
        input_spike_synapse_ids,
        input_spike_sorted_ids,
        output_spike_times,
        output_spike_neuron_ids,
        feedforward_weights,
        recurrent_weights,
        max_input_spikes,
        max_output_spikes,
	n_neurons,
	n_synapses,
        leak_conductance,
        tau_membrane,
        tau_synapse,
        threshold_potential,
        reset_potential,
        refractory_period,
        epsilon
    );

    derivative_evaluator.evaluate_input_spike_time_derivatives(input_spike_id, input_spike_time_derivatives);
}

__global__ void get_feedforward_weight_derivatives_kernel(
    // Input and output arrays
    const float *input_spike_times,
    const int *input_spike_synapse_ids,
    const int *input_spike_sorted_ids,
    const float *output_spike_times,
    const int *output_spike_neuron_ids,
    const float *feedforward_weights,
    const float *recurrent_weights,
    float *feedforward_weight_derivatives,
    // Array dimensions
    int batch_size,
    int max_input_spikes,
    int max_output_spikes,
    int n_neurons,
    int n_synapses,
    // Neuron parameters
    float leak_conductance,
    float tau_membrane,
    float tau_synapse,
    float threshold_potential,
    float reset_potential,
    float refractory_period
) {
    const int global_thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    const int example_id = global_thread_id / (n_neurons * n_synapses);
    const int neuron_id = global_thread_id / n_synapses % n_neurons;
    const int synapse_id = global_thread_id % n_synapses;
    const float epsilon = 1e-6f;

    // Return if this outside the bounds of the problem
    if (example_id >= batch_size || neuron_id >= n_neurons || synapse_id >= n_synapses) {
	return;
    }

    // Look at the parts of the arrays corresponding to this particular example and feedforward weight
    input_spike_times = &input_spike_times[example_id * max_input_spikes];
    input_spike_synapse_ids = &input_spike_synapse_ids[example_id * max_input_spikes];
    input_spike_sorted_ids = &input_spike_sorted_ids[example_id * max_input_spikes];
    output_spike_times = &output_spike_times[example_id * max_output_spikes];
    output_spike_neuron_ids = &output_spike_neuron_ids[example_id * max_output_spikes];
    feedforward_weight_derivatives = &feedforward_weight_derivatives[global_thread_id * max_output_spikes];

    const DerivativeEvaluator derivative_evaluator(
        input_spike_times,
        input_spike_synapse_ids,
        input_spike_sorted_ids,
        output_spike_times,
        output_spike_neuron_ids,
        feedforward_weights,
        recurrent_weights,
        max_input_spikes,
        max_output_spikes,
	n_neurons,
	n_synapses,
        leak_conductance,
        tau_membrane,
        tau_synapse,
        threshold_potential,
        reset_potential,
        refractory_period,
        epsilon
    );

    derivative_evaluator.evaluate_feedforward_weight_derivatives(
        neuron_id, synapse_id, feedforward_weight_derivatives
    );
}

__global__ void get_recurrent_weight_derivatives_kernel(
    // Input and output arrays
    const float *input_spike_times,
    const int *input_spike_synapse_ids,
    const int *input_spike_sorted_ids,
    const float *output_spike_times,
    const int *output_spike_neuron_ids,
    const float *feedforward_weights,
    const float *recurrent_weights,
    float *recurrent_weight_derivatives,
    // Array dimensions
    int batch_size,
    int max_input_spikes,
    int max_output_spikes,
    int n_neurons,
    int n_synapses,
    // Neuron parameters
    float leak_conductance,
    float tau_membrane,
    float tau_synapse,
    float threshold_potential,
    float reset_potential,
    float refractory_period
) {
    const int global_thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    const int example_id = global_thread_id / (n_neurons * n_neurons);
    const int target_neuron_id = global_thread_id / n_neurons % n_neurons;
    const int source_neuron_id = global_thread_id % n_neurons;
    const float epsilon = 1e-6f;

    // Return if this outside the bounds of the problem
    if (example_id >= batch_size || target_neuron_id >= n_neurons || source_neuron_id >= n_neurons) {
	return;
    }

    // Look at the parts of the arrays corresponding to this particular example and recurrent weight
    input_spike_times = &input_spike_times[example_id * max_input_spikes];
    input_spike_synapse_ids = &input_spike_synapse_ids[example_id * max_input_spikes];
    input_spike_sorted_ids = &input_spike_sorted_ids[example_id * max_input_spikes];
    output_spike_times = &output_spike_times[example_id * max_output_spikes];
    output_spike_neuron_ids = &output_spike_neuron_ids[example_id * max_output_spikes];
    recurrent_weight_derivatives = &recurrent_weight_derivatives[global_thread_id * max_output_spikes];

    const DerivativeEvaluator derivative_evaluator(
        input_spike_times,
        input_spike_synapse_ids,
        input_spike_sorted_ids,
        output_spike_times,
        output_spike_neuron_ids,
        feedforward_weights,
        recurrent_weights,
        max_input_spikes,
        max_output_spikes,
	n_neurons,
	n_synapses,
        leak_conductance,
        tau_membrane,
        tau_synapse,
        threshold_potential,
        reset_potential,
        refractory_period,
        epsilon
    );

    derivative_evaluator.evaluate_recurrent_weight_derivatives(
        target_neuron_id, source_neuron_id, recurrent_weight_derivatives
    );
}

size_t divide_and_round_up(size_t numerator, size_t denominator) {
    size_t quotient = numerator / denominator;
    size_t remainder = numerator % denominator;

    if (remainder == 0) {
        return quotient;
    } else {
        return quotient + 1;
    }
}

#define THREADS_PER_BLOCK 512

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
) {
    // Extract the problem shape
    const int batch_size = input_spike_times.size(0);
    const int max_input_spikes = input_spike_times.size(1);
    const int max_output_spikes = output_spike_times.size(1);
    const int n_neurons = feedforward_weights.size(0);
    const int n_synapses = feedforward_weights.size(1);

    // Get the chronologically sorted input spike ids
    torch::Tensor input_spike_sorted_ids = torch::argsort(
        input_spike_times, /*dim=*/1, /*descending=*/false
    ).to(torch::kInt);

    // Create an empty tensor to hold the input spike time derivatives
    torch::Tensor input_spike_time_derivatives = torch::empty(
        {batch_size, max_input_spikes, max_output_spikes},
    	torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA)
    );

    const int n_threads = batch_size * max_input_spikes;
    const int n_blocks = divide_and_round_up(n_threads, THREADS_PER_BLOCK);
    get_input_spike_time_derivatives_kernel<<<n_blocks, THREADS_PER_BLOCK>>>(
        input_spike_times.data<float>(),
        input_spike_synapse_ids.data<int>(),
        input_spike_sorted_ids.data<int>(),
	output_spike_times.data<float>(),
	output_spike_neuron_ids.data<int>(),
        feedforward_weights.data<float>(),
        recurrent_weights.data<float>(),
	input_spike_time_derivatives.data<float>(),
        batch_size,
        max_input_spikes,
	max_output_spikes,
        n_neurons,
        n_synapses,
        leak_conductance,
        tau_membrane,
        tau_synapse,
        threshold_potential,
        reset_potential,
        refractory_period
    );

    // Swap the final two axes of the derivative array to get the
    // final shape: (batch_size, max_output_spikes, max_input_spikes)
    return input_spike_time_derivatives.transpose(1, 2);
}

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
) {
    // Extract the problem shape
    const int batch_size = input_spike_times.size(0);
    const int max_input_spikes = input_spike_times.size(1);
    const int max_output_spikes = output_spike_times.size(1);
    const int n_neurons = feedforward_weights.size(0);
    const int n_synapses = feedforward_weights.size(1);

    // Get the chronologically sorted input spike ids
    torch::Tensor input_spike_sorted_ids = torch::argsort(
        input_spike_times, /*dim=*/1, /*descending=*/false
    ).to(torch::kInt);

    // Create an empty tensor to hold the feedforward weight derivatives
    torch::Tensor feedforward_weight_derivatives = torch::empty(
        {batch_size, n_neurons, n_synapses, max_output_spikes},
	torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA)
    );

    const int n_threads = batch_size * n_neurons * n_synapses;
    const int n_blocks = divide_and_round_up(n_threads, THREADS_PER_BLOCK);
    get_feedforward_weight_derivatives_kernel<<<n_blocks, THREADS_PER_BLOCK>>>(
        input_spike_times.data<float>(),
        input_spike_synapse_ids.data<int>(),
        input_spike_sorted_ids.data<int>(),
	output_spike_times.data<float>(),
	output_spike_neuron_ids.data<int>(),
        feedforward_weights.data<float>(),
        recurrent_weights.data<float>(),
	feedforward_weight_derivatives.data<float>(),
        batch_size,
        max_input_spikes,
	max_output_spikes,
        n_neurons,
        n_synapses,
        leak_conductance,
        tau_membrane,
        tau_synapse,
        threshold_potential,
        reset_potential,
        refractory_period
    );

    // Permute the axes of the derivative array to get the final
    // shape: (batch_size, max_output_spikes, n_neurons, n_synapses)
    return feedforward_weight_derivatives.permute({0, 3, 1, 2});
}

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
) {
    // Extract the problem shape
    const int batch_size = input_spike_times.size(0);
    const int max_input_spikes = input_spike_times.size(1);
    const int max_output_spikes = output_spike_times.size(1);
    const int n_neurons = feedforward_weights.size(0);
    const int n_synapses = feedforward_weights.size(1);

    // Get the chronologically sorted input spike ids
    torch::Tensor input_spike_sorted_ids = torch::argsort(
        input_spike_times, /*dim=*/1, /*descending=*/false
    ).to(torch::kInt);

    // Create an empty tensor to hold the recurrent weight derivatives
    torch::Tensor recurrent_weight_derivatives = torch::empty(
        {batch_size, n_neurons, n_neurons, max_output_spikes},
    	torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA)
    );

    const int n_threads = batch_size * n_neurons * n_neurons;
    const int n_blocks = divide_and_round_up(n_threads, THREADS_PER_BLOCK);
    get_recurrent_weight_derivatives_kernel<<<n_blocks, THREADS_PER_BLOCK>>>(
        input_spike_times.data<float>(),
        input_spike_synapse_ids.data<int>(),
        input_spike_sorted_ids.data<int>(),
	output_spike_times.data<float>(),
	output_spike_neuron_ids.data<int>(),
        feedforward_weights.data<float>(),
        recurrent_weights.data<float>(),
	recurrent_weight_derivatives.data<float>(),
        batch_size,
        max_input_spikes,
	max_output_spikes,
        n_neurons,
        n_synapses,
        leak_conductance,
        tau_membrane,
        tau_synapse,
        threshold_potential,
        reset_potential,
        refractory_period
    );

    // Permute the axes of the derivative array to get the final
    // shape: (batch_size, max_output_spikes, n_neurons, n_neurons)
    return recurrent_weight_derivatives.permute({0, 3, 1, 2});
}
