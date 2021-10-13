import csv
import numpy as np


def get_speed_stats(csv_path):



	row_num = 1
	veh_id_nums = 0

	curr_veh_id = 'id'
	speeds = []

	results = []

	with open(csv_path, newline='') as csvfile:
		
		csvreader = csv.reader(csvfile, delimiter=',')

		for row in csvreader:

			if(row_num > 1):
				# Don't read header

				if(curr_veh_id != row[1] and row_num > 2):
					#Switched to a new veh_id:

					veh_id_nums += 1
					if(len(speeds) > 0):
						results.append([curr_veh_id,np.mean(speeds),np.sqrt(np.var(speeds))])
						sys.stdout.write('\r'+'Veh id: '+str(veh_id_nums))

					speeds = [] # Reset speeds
					curr_veh_id = row[1] # Set new veh id

				# Add data from new row:
				# sys.stdout.write('\r'+'Veh id: '+str(veh_id_nums) + ' row: ' +str(row_num))

				curr_veh_id = row[1]
				time = float(row[0])
				edge = row[13]

				include_data = (time>300 and edge != 'Eastbound_On_1' and edge != 'Eastbound_Off_2')

				if(include_data):

					# sys.stdout.write('\r'+'Veh id: '+str(veh_id_nums) + ' row: ' +str(row_num))
					speed = float(row[4])
					speeds.append(speed)

			row_num += 1 #Keep track of which row is being looked at

	return results	







