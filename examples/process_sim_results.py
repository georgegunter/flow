import numpy as np
import matplotlib.pyplot


def process_sim_result(sim_results):
	#This is hardcoded with assumptions about the structure
	#of what's in each column. If that changes, need to change
	#here too.

	attack_durations = np.unique(sim_results[:,0])
	attack_magnitudes = np.unique(sim_results[:,1])

	attack_durations_list = list(attack_durations)
	attack_magnitudes_list = list(attack_durations)

	num_results = len(sim_results[:,0])

	fuel_rates = np.zeros((len(attack_durations),len(attack_magnitudes)))

	travel_times = np.zeros((len(attack_durations),len(attack_magnitudes)))

	speed_vars = np.zeros((len(attack_durations),len(attack_magnitudes)))

	num_vehicles = np.zeros((len(attack_durations),len(attack_magnitudes)))

	for n in range(num_results):
		attack_dur = sim_results[n,0]
		attack_mag = sim_results[n,1]
		attack_dur_index = attack_durations_list.index(attack_dur)
		attack_mag_index = attack_magnitudes_list.index(attack_mag)

		fuel_rate = sim_results[n,6]/sim_results[n,5]
		travel_time = sim_results[n,7]
		speed_var = sim_results[n,8]
		num_vehicle = sim_results[n,9]

		fuel_rates[attack_dur_index,attack_mag_index] = fuel_rate

		travel_times[attack_dur_index,attack_mag_index] = travel_time

		speed_vars[attack_dur_index,attack_mag_index] = speed_var

		num_vehicles[attack_dur_index,attack_mag_index] = num_vehicle

		return [attack_durations,attack_magnitudes,fuel_rates,travel_times,speed_vars,num_vehicles]









