function out = exponential_decay(learning_rate, global_step, decay_steps, decay_rate, staircase)

out = learning_rate*decay_rate^(global_step / decay_steps);

end