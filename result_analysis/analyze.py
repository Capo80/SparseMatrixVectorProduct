import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
import csv

MATRICES_PATH = "../matrices"
MAX_THREADS = 40
def get_metrix_size(matrix_dict, matrix_name):
	if (matrix_name in matrix_dict["small"]):
		return "small"
	if (matrix_name in matrix_dict["medium"]):
		return "medium"
	if (matrix_name in matrix_dict["big"]):
		return "big"

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
	if (matrix[2] < 4000):	
		matrix_info["small"].append(name)
	elif (matrix[2] < 400000):	
		matrix_info["medium"].append(name)
	else:	
		matrix_info["big"].append(name)

print(matrix_info)

#read result file
csvfile = open('results.csv', newline='')

reader = csv.DictReader(csvfile)


#Analyze number of threads
threads_x = [i for i in range(1, MAX_THREADS+1)]
matrix_times_threads = {"small": [0]*MAX_THREADS, "medium": [0]*MAX_THREADS, "big": [0]*MAX_THREADS}
matrix_times_threads2 = { "small": [0]*MAX_THREADS, "medium": [0]*MAX_THREADS, "big": [0]*MAX_THREADS}

formats = { "csr" : matrix_times_threads, "ellpack": matrix_times_threads2}


for row in reader:
	#only omp
	if (int(row["product"]) == 2):
		#find correct format
		if (int(row["format"]) == 1):
			format_name = "ellpack"
		else:
			format_name = "csr"

		c = 0

		thread = int(row["threads"])-1

		#sum matrix times in their category
		for name, time in row.items():
			if (name != "product" and name != "format" and name != "threads"):
				formats[format_name][get_metrix_size(matrix_info, name)][thread] += float(time)
				c += 1
		#calculate averages
		formats[format_name]["small"][thread] /= c
		formats[format_name]["medium"][thread] /= c
		formats[format_name]["big"][thread] /= c
			
plt.figure(0)

plt.plot(threads_x, formats["csr"]["small"], label="small")
plt.plot(threads_x, formats["csr"]["medium"], label="medium")
plt.plot(threads_x, formats["csr"]["big"], label="big")

plt.figure(1)

plt.plot(threads_x, formats["ellpack"]["small"], label="small")
plt.plot(threads_x, formats["ellpack"]["medium"], label="medium")
plt.plot(threads_x, formats["ellpack"]["big"], label="big")

plt.legend()
plt.show()