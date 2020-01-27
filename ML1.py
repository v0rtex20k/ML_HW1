import numpy as np
import math

# For 3D Arrays: (HEIGHT, WIDTH, DEPTH)
def calc_k_nearest_neighbors(data_NF, query_QF, K=1):
	the_hood = np.zeros((query_QF.shape[0], K, query_QF.shape[1]))
	for query in range(query_QF.shape[0]):
		distances = [(0,0) for x in range(data_NF.shape[0])]
		for instance in range(data_NF.shape[0]):
			d = euclidean_distance(query_QF[query], data_NF[instance], query_QF.shape[1])
			distances[instance] = (d,instance)
		distances = sorted(distances, key= lambda x:x[0])
		for i in range(K):
			neighbor_row = data_NF[distances[i][1]] # now an F dimensional 1D array
			the_hood[query][i] = neighbor_row
	return the_hood

# shape reported as (num_rows, num_cols)
def split_into_train_and_test(x_all_LF, frac_test=0.5, random_state=None):
	num_training_instances = int((1 - frac_test) * x_all_LF.shape[0])  # floors it
	training_set = np.zeros((num_training_instances, x_all_LF.shape[1]))
	test_set = np.zeros(((x_all_LF.shape[0] - num_training_instances), x_all_LF.shape[1]))
	for feature in range(num_training_instances):
		training_row = random_state.randint(0, high= x_all_LF.shape[0])
		training_set[feature] = x_all_LF[training_row]
		x_all_LF = np.delete(x_all_LF, training_row, axis=0)
	test_set = x_all_LF.copy()
	return (training_set, test_set)

# inputs are f-dimensional vectors
def euclidean_distance(x1, x2, f):
	sum = 0
	for i in range(f):
		sum += pow((x1[i] - x2[i]), 2)
	return math.sqrt(sum)

def output1(train, test):
	print(train.shape)
	print(test.shape)
	print(train)
	print(test)

def output2(hood):
	print(hood.shape)
	print(hood)

if __name__ == '__main__':
	x_LF = np.eye(10)
	xcopy_LF = x_LF.copy()
	train_MF, test_NF = split_into_train_and_test(x_LF, frac_test=0.3, random_state=np.random.RandomState(0))
	output1(train_MF, test_NF)
	print(np.allclose(x_LF, xcopy_LF))
	neighb_QKF = calc_k_nearest_neighbors(train_MF, test_NF)
	output2(neighb_QKF)


