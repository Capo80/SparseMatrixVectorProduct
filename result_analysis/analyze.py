import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
import csv
import numpy as np

MATRICES_PATH = "../matrices"
RESULTS_PATH = "../results.csv"
MAX_THREADS = 40

def max_non_zeros(matrix_name):
	file = open(join(MATRICES_PATH, matrix_name+ ".mtx"), "r")

	first_line = True
	for line in file:
		if line[0] != "%":
			if first_line:
				sizes = [int(i.strip()) for i in line.split(" ")]
				first_line = False
				lines_zeros = [0]*sizes[0]
			else:
				lines_zeros[int(line.split(" ")[0])-1] += 1


	file.close()
	return np.max(lines_zeros)
def get_metrix_size(matrix_dict, matrix_name):
	if (matrix_name in matrix_dict["small"]):
		return "small"
	if (matrix_name in matrix_dict["medium"]):
		return "medium"
	if (matrix_name in matrix_dict["big"]):
		return "big"

def get_threads_graph():
	#Analyze number of threads
	threads_x = [i for i in range(1, MAX_THREADS+1)]

	formats = { "csr" : {}, "ellpack": {}}

	for row in reader:
		#only omp
		if (int(row["product"]) == 2):
			#find correct format
			if (int(row["format"]) == 1):
				format_name = "ellpack"
			else:
				format_name = "csr"

			thread = int(row["threads"])-1

			#sum matrix times in their category
			for name, time in row.items():
				if (name != "product" and name != "format" and name != "threads"):
					if (formats[format_name].get(name) == None):
						formats[format_name][name] = [float(time)]
					else:
						formats[format_name][name].append(float(time))
						
					
	#plot threads
	plt.figure(0)

	for name, value in formats["csr"].items():
		if (name in ["mcfe.mtx", "adder_dcop_32.mtx", "webbase-1M.mtx", "ML_Laplace.mtx", "bcsstk17.mtx"]):					
			plt.plot(threads_x, value, label=name)

	plt.xlabel("Threads")
	plt.ylabel("FLOPS")
	plt.legend()

	plt.figure(1)

	for name, value in formats["ellpack"].items():
		if (name in ["mcfe.mtx", "adder_dcop_32.mtx", "webbase-1M.mtx", "ML_Laplace.mtx", "bcsstk17.mtx", "thermal2.mtx", "raefsky2.mtx"]):					
			plt.plot(threads_x, value, label=name)

	plt.xlabel("Threads")
	plt.ylabel("FLOPS")
	plt.legend()
	plt.show()


def bar_performace(product_number):
	# analyze serial performance
	sizes = [i for i in matrix_info.keys()]
	names = []
	times1 = []
	times2 = []
	matrix_serial1 = {"small": 0, "medium": 0, "big": 0}
	matrix_serial2 = {"small": 0, "medium": 0, "big": 0}
	formats = {"csr" : times1, "ellpack": times2}

	for row in reader:
		if (int(row["product"]) == product_number):
			#find correct format
			if (int(row["format"]) == 1):
				format_name = "ellpack"
			else:
				format_name = "csr"
			
			#sum matrix times in their category
			for name, time in row.items():
				if (name != "product" and name != "format" and name != "threads"):
					#formats[format_name][get_metrix_size(matrix_info, name)] += float(time)
					formats[format_name].append(float(time))
					if name not in names:	
						names.append(name)

#			print(formats.keys())
#			print(sizes)
#			for i in formats.keys():
#				for j in matrix_serial1.keys():
#					formats[i][j] /= len(matrix_info[j])
#					print(i, j)

	#bar plot
	plt.figure(0)


	plt.bar(names, formats["csr"])

	plt.xlabel("Matrix size")
	plt.ylabel("FLOPS")
	plt.legend()

	plt.figure(1)

	plt.bar(names, formats["ellpack"])

	plt.xlabel("Matrix size")
	plt.ylabel("FLOPS")
	plt.legend()
	plt.show()

def calculate_speed_up():

	names = [f.split(".")[0] for f in listdir(MATRICES_PATH) if isfile(join(MATRICES_PATH, f))]
	times_cpu1 = [0]*len(names)
	times_cpu2 = [0]*len(names)
	times_gpu1 = [0]*len(names)
	times_gpu2 = [0]*len(names)
	formats = {"csr" : (times_cpu1, times_gpu1), "ellpack": (times_cpu2, times_gpu2)}
	
	go_in = -1
	table = [[[0] for i in range(0, 5)] for j in range(0, len(names))]
	for row in reader:
		#print(row)
		if (int(row["product"]) == 2 and int(row["threads"]) == 20):
			go_in = 0
		elif (int(row["product"]) == 3):
			go_in = 1

		if (go_in != -1):

			if (int(row["format"]) == 1):
				format_name = "ellpack"
			else:
				format_name = "csr"

			for name, time in row.items():
				if (name != "product" and name != "format" and name != "threads"):
					formats[format_name][go_in][names.index(name.split(".")[0])] = float(time)
					table[names.index(name.split(".")[0])][0] = name.split(".")[0]
					if (go_in == 0 and format_name == "csr"):
						table[names.index(name.split(".")[0])][1] = float(time)
					elif (go_in == 0 and format_name == "ellpack"):
						table[names.index(name.split(".")[0])][2] = float(time)
					elif (go_in == 1 and format_name == "csr"):
						table[names.index(name.split(".")[0])][3] = float(time)
					elif (go_in == 1 and format_name == "ellpack"):
						table[names.index(name.split(".")[0])][4] = float(time)


			go_in = -1
	
	print(table)
	for line in table:
		if line[2] != -1:
			print(line[0].replace("_", "\\_"), "&", '{:.2f}'.format(line[1] / 10**9), "&", '{:.2f}'.format(line[2]/ 10**9), "&", '{:.2f}'.format(line[3]/ 10**9), "&", '{:.2f}'.format(line[4]/ 10**9), "\\\\")
		else:
			print(line[0].replace("_", "\\_"), "&", '{:.2f}'.format(line[1]/ 10**9), "& na &", '{:.2f}'.format(line[3]/ 10**9), "& na \\\\")
		
		print("\\hline")
	plt.figure(0)

	threshold = 1.0
	values = np.array([i/j for (i,j) in zip(formats["csr"][1],formats["csr"][0])])
	above_threshold = [0 if a < threshold else a for a in values]
	below_threshold = [0 if a >= threshold else a for a in values]

	plt.bar(names, below_threshold, color="red")
	plt.bar(names, above_threshold, color="green")


	plt.axhline(y=threshold,linewidth=1, color='k')

	plt.xlabel("Matrix name")
	plt.ylabel("Speed-up")

	y_pos = range(len(names))
	plt.xticks(y_pos, names, rotation='vertical')

	print(np.mean(values))
	plt.figure(1)

	threshold = 1.0
	values = np.array([i/j for (i,j) in zip(formats["ellpack"][1],formats["ellpack"][0])])
	above_threshold = [0 if a < threshold else a for a in values]
	below_threshold = [0 if a >= threshold else a for a in values]

	plt.bar(names, below_threshold, color="red")
	plt.bar(names, above_threshold, color="green")


	plt.axhline(y=threshold,linewidth=1, color='k')

	plt.xlabel("Matrix name")
	plt.ylabel("Speed-up")
	y_pos = range(len(names))
	plt.xticks(y_pos, names, rotation='vertical')

	print(np.mean(values))
	plt.show()

def check_formats():

	names = [f.split(".")[0] for f in listdir(MATRICES_PATH) if isfile(join(MATRICES_PATH, f))]
	times_gpu1 = [0]*len(names)
	times_gpu2 = [0]*len(names)
	formats = {"csr" : times_gpu1, "ellpack": times_gpu2}
	
	go_in = -1
	for row in reader:
		#print(row)
		if (int(row["product"]) == 3):
			go_in = 1

		if (go_in != -1):

			if (int(row["format"]) == 1):
				format_name = "ellpack"
			else:
				format_name = "csr"

			for name, time in row.items():
				if (name != "product" and name != "format" and name != "threads"):
					formats[format_name][names.index(name.split(".")[0])] = float(time)

			go_in = -1
	
	threshold = 1
	to_remove = []
	for name in list(names):
		ratio = matrix_sizes[name+ ".mtx"][0]*max_non_zeros(name) * 1.0 / matrix_sizes[name+ ".mtx"][2]
		print(name, "&", "{:.4f}".format(formats["ellpack"][names.index(name)] / formats["csr"][names.index(name)]), "&", "{:.4f}".format(ratio), "\\\\")
		print("\\hline")
		if ratio > 1.5:
			new_index = names.index(name)
			names.remove(names[new_index])
			formats["ellpack"].remove(formats["ellpack"][new_index])
			formats["csr"].remove(formats["csr"][new_index])
	
	values = np.array([i/j for (i,j) in zip(formats["ellpack"],formats["csr"])])
	above_threshold = [0 if a < threshold else a for a in values]
	below_threshold = [0 if a >= threshold else a for a in values]

	plt.bar(names, below_threshold, color="red")
	plt.bar(names, above_threshold, color="green")

	plt.axhline(y=threshold,linewidth=1, color='k')

	plt.xlabel("Matrix name")
	plt.ylabel("FLOPS")
	y_pos = range(len(names))
	plt.xticks(y_pos, names, rotation='vertical')

	plt.show()

#read all matrix dimensions
matrices = [f for f in listdir(MATRICES_PATH) if isfile(join(MATRICES_PATH, f))]

matrix_sizes = {}
for file in matrices:
	for line in open(join(MATRICES_PATH, file), "r"):
		if line[0] != "%":
			dimensions = [int(i) for i in line.strip().split(" ")]
			matrix_sizes[file] = dimensions
			break

#separate different sizes
matrix_info = {"small": [], "medium": [], "big": []}

for name, matrix in matrix_sizes.items():
	if (matrix[0] < 5000):	
		matrix_info["small"].append(name)
	elif (matrix[0] < 500000):	
		matrix_info["medium"].append(name)
	else:	
		matrix_info["big"].append(name)

#print(matrix_info)

#read result file
csvfile = open(RESULTS_PATH, newline='')

reader = csv.DictReader(csvfile)

print(check_formats())
