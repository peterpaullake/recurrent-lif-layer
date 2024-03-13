#include <torch/extension.h>

class LIFNeuron;

class SpikeHistory {
    /**
     * The `SpikeHistory` class provides an interface for storing the effect
     * of the previous input and output spikes on the membrane dynamics.
     * 
     * The spike times and weights themselves aren't stored. Instead, their
     * contributions to the corresponding terms of the membrane potential
     * u(t) are accumulated in the member variables `m_membrane_history`
     * and `m_synapse_history`. This allows us to avoid summing over all
     * the previous spikes every time we evaluate the membrane potential.
     **/

private:
    float m_membrane_history;
    float m_synapse_history;
    float m_latest_recorded_spike_time;
    const LIFNeuron *m_lif_neuron;

public:
    __device__ SpikeHistory(const LIFNeuron *lif_neuron);

    /**
     * Updates the membrane and synapse history to account for a new spike.
     **/
    __device__ void record_spike(float spike_time, float weight);

    /**
     * Updates the membrane and synapse history to account for a range of new input spikes.
     *
     * Parameters:
     *   - input_spike_times: an array of input spike times
     *   - input_spike_synapse_ids: the synapse ids associated with each input spike
     *   - input_spike_sorted_ids: indices that would sort the input spikes chronologically
     *   - feedforward_weights: the feedforward synaptic weight values
     *   - n_skip_input_spikes: the number of initial input spikes to skip over
     *   - max_input_spikes: the number of input spikes
     *   - t_stop: the time after which to stop recording input spikes
     **/
    __device__ int record_input_spikes(
        const float *input_spike_times,
	const int *input_spike_synapse_ids,
	const int *input_spike_sorted_ids,
	const float *feedforward_weights,
	int n_skip_input_spikes,
	int max_input_spikes,
	float t_stop
    );

    /**
     * `LIFNeuron` needs to access `m_membrane_history`,
     * `m_synapse_history`, and `m_latest_recorded_spike_time`
     **/
    friend class LIFNeuron;
};

class LIFNeuron {
private:
    float m_leak_conductance;
    float m_tau_membrane;
    float m_tau_synapse;
    float m_threshold_potential;
    float m_prefactor;
    
    __device__ float get_output_spike_time_upper_bound(
        const SpikeHistory *spike_history, float initial_condition_factor
    ) const {
	float membrane_factor = m_prefactor * spike_history->m_membrane_history + initial_condition_factor;
	float synapse_factor = m_prefactor * spike_history->m_synapse_history;
	float membrane_upper_bound = m_tau_membrane * logf(2.0f * fabs(membrane_factor) / m_threshold_potential);
	float synapse_upper_bound = m_tau_synapse * logf(2.0f * fabs(synapse_factor) / m_threshold_potential);
	return fmaxf(membrane_upper_bound, synapse_upper_bound);
    }

    __device__ float evaluate_free_membrane_potential(const SpikeHistory *spike_history, float t) const {
	return evaluate_free_membrane_potential(spike_history, 0.0f, t);
    }

    __device__ float evaluate_free_membrane_potential(
        const SpikeHistory *spike_history, float initial_condition_factor, float t
    ) const {
	// We can only evaluate the membrane potential at or after the last spike recorded
	assert(
            isnan(spike_history->m_latest_recorded_spike_time)
	    || t >= spike_history->m_latest_recorded_spike_time
        );

	float potential = 0.0f;
	potential += m_prefactor * spike_history->m_membrane_history * expf(-t / m_tau_membrane);
	potential += initial_condition_factor * expf(-t / m_tau_membrane);
	potential -= m_prefactor * spike_history->m_synapse_history * expf(-t / m_tau_synapse);
	return potential;
    }
    
    __device__ float get_initial_condition_factor(const SpikeHistory *spike_history, float t0, float u0) const {
	return (u0 - evaluate_free_membrane_potential(spike_history, t0)) * expf(t0 / m_tau_membrane);
    }

    __device__ bool is_free_membrane_potential_above_threshold(
        const SpikeHistory *spike_history, float initial_condition_factor, float t_start, float t_end
    ) const {
	if (evaluate_free_membrane_potential(
	        spike_history, initial_condition_factor, t_start
            ) >= m_threshold_potential
        ) {
	    return true;
	} else if (
	    evaluate_free_membrane_potential(
	        spike_history, initial_condition_factor, t_end
            ) >= m_threshold_potential
        ) {
	    return true;
	} else {
	    // Check the stationary point
	    float membrane_factor = m_prefactor * spike_history->m_membrane_history + initial_condition_factor;
	    float synapse_factor = m_prefactor * spike_history->m_synapse_history;
	    float t_stationary = (
		(logf(membrane_factor / m_tau_membrane) - logf(synapse_factor / m_tau_synapse))
		/ (1.0f / m_tau_membrane - 1.0f / m_tau_synapse)
	    );
	    return (
		!isnan(t_stationary)
		&& t_start <= t_stationary
		&& t_stationary <= t_end
		&& evaluate_free_membrane_potential(
		    spike_history, initial_condition_factor, t_stationary
		) >= m_threshold_potential
	    );
	}
    }

public:
    __device__ LIFNeuron(float leak_conductance, float tau_membrane, float tau_synapse, float threshold_potential)
	: m_leak_conductance(leak_conductance),
	  m_tau_membrane(tau_membrane),
	  m_tau_synapse(tau_synapse),
	  m_threshold_potential(threshold_potential) {
	float membrane_capacitance = m_leak_conductance * m_tau_membrane;
	m_prefactor = (
            1.0f
	    / membrane_capacitance
	    * m_tau_membrane * m_tau_synapse
	    / (m_tau_membrane - m_tau_synapse)
        );
    }

    __device__ float get_next_output_spike_time(
        const SpikeHistory *spike_history,
	float t_start,
	float u_start,
	float *initial_condition_factor,
	const float *input_spike_times,
	const int *input_spike_synapse_ids,
	const int *input_spike_sorted_ids,
	const float *feedforward_weights,
	int n_skip_input_spikes,
	int max_input_spikes,
	float epsilon
    ) const {
	*initial_condition_factor = get_initial_condition_factor(spike_history, t_start, u_start);
	return get_next_output_spike_time(
            spike_history,
	    t_start,
            *initial_condition_factor,
            input_spike_times,
            input_spike_synapse_ids,
	    input_spike_sorted_ids,
            feedforward_weights,
            n_skip_input_spikes,
            max_input_spikes,
            epsilon
        );
    }

    __device__ float get_next_output_spike_time(
        const SpikeHistory *spike_history,
	float t_start,
	float initial_condition_factor,
	const float *input_spike_times,
	const int *input_spike_synapse_ids,
	const int *input_spike_sorted_ids,
	const float *feedforward_weights,
	int n_skip_input_spikes,
	int max_input_spikes,
	float epsilon
    ) const {
	// Make sure `t_start` is not after the earliest input spike to be processed
	if (n_skip_input_spikes < max_input_spikes &&
	    t_start > input_spike_times[input_spike_sorted_ids[n_skip_input_spikes]]) {
	    return NAN;
	}

	// Create a temporary spike history object to record
	// the effect of the input spikes as we process them
	SpikeHistory speculative_spike_history(*spike_history);

	// Loop through each inter-spike interval in chronological order
	int input_spike_id = n_skip_input_spikes;
	do {
	    // End the search interval at the next input spike time. If
	    // the next spike time doesn't exist or is infinite, end the
	    // search interval at the upper bound for the output spike time.
	    float t_end;
	    bool processed_all_input_spikes = (
                // Next spike doesn't exist
                input_spike_id >= max_input_spikes
		// Next spike is infinite
		|| isinf(input_spike_times[input_spike_sorted_ids[input_spike_id]])
            );
	    if (processed_all_input_spikes) {
		t_end = get_output_spike_time_upper_bound(&speculative_spike_history, initial_condition_factor);

		// If the output spike time upper bound is in the past, we know that
		// the neuron won't fire any more, so we can terminate the search
		if (t_end < t_start) {
		    break;
		}
	    } else {
		t_end = input_spike_times[input_spike_sorted_ids[input_spike_id]];
	    }

	    // Make sure the search interval is valid
	    assert(t_start <= t_end);

	    // Check whether the membrane potential is ever at or above
	    // the threshold potential inside the search interval
	    if (is_free_membrane_potential_above_threshold(
		    &speculative_spike_history, initial_condition_factor, t_start, t_end
	    )) {
		// Repeatedly halve the search interval to find, within
		// the tolerance `epsilon`, the earliest time where the
		// membrane potential is at or above the threshold
		int n_halvings = (int) ceil(log2f((t_end - t_start) / epsilon));
		for (int halving_id = 0; halving_id < n_halvings; halving_id++) {
		    float t_mid = (t_start + t_end) / 2.0f;
		    if (is_free_membrane_potential_above_threshold(
                            &speculative_spike_history, initial_condition_factor, t_start, t_mid
                    )) {
			t_end = t_mid;
		    } else {
			t_start = t_mid;
		    }
		}
		return (t_start + t_end) / 2.0f;
	    }

	    // If we've processed all the input spikes and not
	    // found an output spike, we can terminate the search
	    if (processed_all_input_spikes) {
		break;
	    }

	    // Take into account the present input spike when calculating
	    // the membrane potential inside the next inter-spike interval
	    float weight = feedforward_weights[input_spike_synapse_ids[input_spike_sorted_ids[input_spike_id]]];
	    speculative_spike_history.record_spike(
                input_spike_times[input_spike_sorted_ids[input_spike_id]], weight
            );

	    // Start searching for the next output spike starting at `t_end`
	    t_start = t_end;

	    input_spike_id++;
	} while (input_spike_id < max_input_spikes + 1);

	// No output spike was found, so we return positive infinity
	return INFINITY;
    }

    /**
     * `SpikeHistory` needs to access `m_tau_membrane` and `m_tau_synapse`
     */
    friend class SpikeHistory;
};

__device__ SpikeHistory::SpikeHistory(const LIFNeuron *lif_neuron)
    : m_membrane_history(0.0f),
      m_synapse_history(0.0f),
      m_latest_recorded_spike_time(NAN),
      m_lif_neuron(lif_neuron) {}

__device__ void SpikeHistory::record_spike(float spike_time, float weight) {
    m_membrane_history += weight * expf(spike_time / m_lif_neuron->m_tau_membrane);
    m_synapse_history += weight * expf(spike_time / m_lif_neuron->m_tau_synapse);
    m_latest_recorded_spike_time = fmaxf(m_latest_recorded_spike_time, spike_time);
}

__device__ int SpikeHistory::record_input_spikes(
    const float *input_spike_times,
    const int *input_spike_synapse_ids,
    const int *input_spike_sorted_ids,
    const float *feedforward_weights,
    int n_skip_input_spikes,
    int max_input_spikes,
    float t_stop
) {
    int input_spike_id;
    for (input_spike_id = n_skip_input_spikes; input_spike_id < max_input_spikes; input_spike_id++) {
	float input_spike_time = input_spike_times[input_spike_sorted_ids[input_spike_id]];

	if (input_spike_time > t_stop) {
	    break;
	}

	record_spike(
	    input_spike_time, feedforward_weights[input_spike_synapse_ids[input_spike_sorted_ids[input_spike_id]]]
	);
    }

    // Return the number of input spikes processed
    return input_spike_id - n_skip_input_spikes;
}

__device__ float get_min(const float *x, int n) {
    assert(n >= 1);

    float min = x[0];
    for (int i = 1; i < n; i++) {
	if (x[i] < min) {
	    min = x[i];
	}
    }
    return min;
}

__global__ void get_output_spike_times_kernel(
    // Input and output arrays
    const float *input_spike_times,
    const int *input_spike_synapse_ids,
    const int *input_spike_sorted_ids,
    const float *feedforward_weights,
    const float *recurrent_weights,
    float *output_spike_times,
    int *output_spike_neuron_ids,
    // Array dimensions
    int batch_size,
    int max_input_spikes,
    int n_neurons,
    int n_synapses,
    int max_output_spikes,
    // Neuron parameters
    float leak_conductance,
    float tau_membrane,
    float tau_synapse,
    float threshold_potential,
    float reset_potential,
    float refractory_period
) {
    const int example_id = blockIdx.x;
    const int neuron_id = threadIdx.x;
    const float epsilon = 1e-6f;
    const int MAX_NEURONS = 256;

    assert(n_neurons <= MAX_NEURONS);

    // Assert this thread is not outside the bounds of the problem
    assert(example_id < batch_size && neuron_id < n_neurons);

    // Look at the parts of the arrays corresponding to this particular example and neuron
    input_spike_times = &input_spike_times[example_id * max_input_spikes];
    input_spike_synapse_ids = &input_spike_synapse_ids[example_id * max_input_spikes];
    input_spike_sorted_ids = &input_spike_sorted_ids[example_id * max_input_spikes];
    feedforward_weights = &feedforward_weights[neuron_id * n_synapses];
    recurrent_weights = &recurrent_weights[neuron_id * n_neurons];
    output_spike_times = &output_spike_times[example_id * max_output_spikes];
    output_spike_neuron_ids = &output_spike_neuron_ids[example_id * max_output_spikes];

    const LIFNeuron lif_neuron(leak_conductance, tau_membrane, tau_synapse, threshold_potential);

    // Here we declare and initialize the state of the simulation.
    // `spike_history`, `initial_condition_factor`, and `prev_output_spike_time`
    // are specific to this particular combination of example and neuron,
    // but the remaining variables are the same across all the neurons.
    SpikeHistory spike_history(&lif_neuron);
    float time = get_min(input_spike_times, max_input_spikes);
    float initial_condition_factor = 0.0f;
    int n_processed_input_spikes = 0;
    float prev_output_spike_time = NAN;

    int output_spike_id;
    for (output_spike_id = 0; output_spike_id < max_output_spikes; output_spike_id++) {
	// Solve for the next output spike time of this particular example and
	// neuron. If the neuron is currently refractory, we need to start the
	// search immediately after the refractory period, but incorporate
	// any input spikes that happen during the refractory period.
	float next_output_spike_time;
	float refractory_period_remaining = refractory_period - (time - prev_output_spike_time);
	bool is_refractory = !isnan(refractory_period_remaining) && refractory_period_remaining >= 0.0f;
	if (is_refractory) {
	    // Deal with the refractory period by temporarily (only within the
	    // present scope and not affecting the neuron's state) processing the
	    // input spikes during the refractory period and then starting the
	    // search for the next output spike after the refractory period
	    SpikeHistory speculative_spike_history(spike_history);
	    float refractory_period_end_time = time + refractory_period_remaining;
	    int n_processed_refractory_input_spikes = speculative_spike_history.record_input_spikes(
                input_spike_times,
		input_spike_synapse_ids,
		input_spike_sorted_ids,
		feedforward_weights,
		n_processed_input_spikes,
		max_input_spikes,
		refractory_period_end_time
	    );

	    // Solve for the next output spike time after the refractory period
	    next_output_spike_time = lif_neuron.get_next_output_spike_time(
	        &speculative_spike_history,
		refractory_period_end_time,
		reset_potential,
		&initial_condition_factor,
		input_spike_times,
		input_spike_synapse_ids,
		input_spike_sorted_ids,
		feedforward_weights,
		n_processed_input_spikes + n_processed_refractory_input_spikes,
		max_input_spikes,
		epsilon
            );
	} else {
	    // Solve for the next output spike time
	    next_output_spike_time = lif_neuron.get_next_output_spike_time(
	        &spike_history,
		time,
		initial_condition_factor,
		input_spike_times,
		input_spike_synapse_ids,
		input_spike_sorted_ids,
		feedforward_weights,
		n_processed_input_spikes,
		max_input_spikes,
		epsilon
            );
	}

	// Store the next output spike time of every neuron on this example
	// https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory
	__syncthreads();
	__shared__ float next_output_spike_times[MAX_NEURONS];
	next_output_spike_times[neuron_id] = next_output_spike_time;
	__syncthreads();
	
	// Find the earliest output spike time of all the neurons for this example
	next_output_spike_time = next_output_spike_times[0];
	int next_output_spike_neuron_id = 0;
	for (int neuron_id2 = 1; neuron_id2 < n_neurons; neuron_id2++) {
	    if (next_output_spike_times[neuron_id2] < next_output_spike_time) {
		next_output_spike_time = next_output_spike_times[neuron_id2];
		next_output_spike_neuron_id = neuron_id2;
	    }
	}

	// Quit if no neuron spiked
	if (isinf(next_output_spike_time)) {
	    break;
	}

	// Save the output spike
	if (neuron_id == 0) { // ensure that only one thread per block runs this
	    output_spike_times[output_spike_id] = next_output_spike_time;
	    output_spike_neuron_ids[output_spike_id] = next_output_spike_neuron_id;
	}

	// Update this neuron's state
	n_processed_input_spikes += spike_history.record_input_spikes(
            input_spike_times,
	    input_spike_synapse_ids,
	    input_spike_sorted_ids,
	    feedforward_weights,
	    n_processed_input_spikes,
	    max_input_spikes,
	    next_output_spike_time
        );
	spike_history.record_spike(
            next_output_spike_time,
            recurrent_weights[next_output_spike_neuron_id]
        );
	time = next_output_spike_time;

	// If it was this neuron that spiked, update the neuron's state accordingly
	if (next_output_spike_neuron_id == neuron_id) {
	    prev_output_spike_time = next_output_spike_time;
	}
    }

    // Set any remaining output spike times to positive infinity so that they're ignored
    if (neuron_id == 0) { // ensure that only one thread per block runs this
	for (; output_spike_id < max_output_spikes; output_spike_id++) {
	    output_spike_times[output_spike_id] = INFINITY;
	}
    }
}

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
) {
    // Extract the problem shape
    const int batch_size = input_spike_times.size(0);
    const int max_input_spikes = input_spike_times.size(1);
    const int n_neurons = feedforward_weights.size(0);
    const int n_synapses = feedforward_weights.size(1);

    // Get the chronologically sorted input spike ids
    torch::Tensor input_spike_sorted_ids = torch::argsort(
        input_spike_times, /*dim=*/1, /*descending=*/false
    ).to(torch::kInt);

    // Create empty tensors to hold the output spikes
    torch::Tensor output_spike_times = torch::empty(
        {batch_size, max_output_spikes},
	torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA)
    );
    torch::Tensor output_spike_neuron_ids = torch::empty(
        {batch_size, max_output_spikes},
	torch::TensorOptions().dtype(torch::kInt).device(torch::kCUDA)
    );

    get_output_spike_times_kernel<<<batch_size, n_neurons>>>(
        input_spike_times.data<float>(),
        input_spike_synapse_ids.data<int>(),
        input_spike_sorted_ids.data<int>(),
        feedforward_weights.data<float>(),
        recurrent_weights.data<float>(),
        output_spike_times.data<float>(),
        output_spike_neuron_ids.data<int>(),
        batch_size,
        max_input_spikes,
        n_neurons,
        n_synapses,
        max_output_spikes,
        leak_conductance,
        tau_membrane,
        tau_synapse,
        threshold_potential,
        reset_potential,
        refractory_period
    );

    return {output_spike_times, output_spike_neuron_ids};
}
