from collections import defaultdict

import networkx as nx
import numpy as np
import scipy
import scipy.sparse.linalg
from scipy.sparse import lil_matrix
from scipy.io import loadmat
import random
import matplotlib.pyplot as plt
import time
import math
from numpy.linalg import matrix_power
import os


class DOULION:

	def __init__(self, G, p):
		'''
		:param G: unweighted graph G(V, E), networkx.Graph()
		:param p: Sparsification parameter p
		'''

		self.G = G
		self.sampled_G = nx.Graph()
		# initiate node attributes
		nx.set_edge_attributes(G, 0, 'w')
		self.p = p

		return

	def run(self, triangle_cnt_alg):
		'''
		:param triangle_cnt_alg:
			"node_iter",
			"edge_iter".
			"trace_exact",
			"trace_est",
			"birthday"
		'''

		for e in G.edges():
			# toss a coin with success prob. p
			if random.random() < self.p:
				self.sampled_G.add_edge(e[0], e[1])
			else:
				pass

		triangle_cnt = self.tri_count(triangle_cnt_alg)

		return triangle_cnt

	def tri_count(self, alg):
		'''
		:param alg:
		'''
		# deterministic
		if alg == "node_iter":
			cnt = self.node_iter()
		elif alg == "edge_iter":
			cnt = self.edge_iter()
		elif alg == "trace_exact":
			cnt = self.trace_exact()
		# randomized
		elif alg == "trace_est":
			cnt = self.trace_est()
		elif alg == "birthday":
			cnt = self.birthday_paradox()
		elif alg == "eigen_est":
			cnt = self.eigen_est()
		elif alg == "":
			pass

		return cnt

	def node_iter(self):
		'''
		node iterator
		use networkx implementation
		'''

		tri_dic = nx.triangles(self.sampled_G) # return tri cnt by node, {n1:t1, n2:t2, ...}
		cnt_total = sum(tri_dic.values()) / 3

		return cnt_total

	def edge_iter(self):
		num_triangles = 0
		for edge in self.sampled_G.edges:
			n = edge[0]
			m = edge[1]
			nodeset_n = set([s for s in self.sampled_G.neighbors(n)])
			nodeset_m = set([s for s in self.sampled_G.neighbors(m)])
			num_triangles += len(nodeset_m.intersection(nodeset_n))

		return num_triangles / 3

	def birthday_paradox(self):
		"""
		implementation of "A space efficient streaming algorithm for estimating
		transitivity and triangle counts using the birthday paradox"

		:return:
		"""

		#se and sw are parameters, 10,000 was used in the paper.

		self.se = 10000
		print(self.se)
		self.edge_res = [ None for i in range(self.se)] # list to store reservoir sample of edges
		self.BirthdayGraph = nx.Graph() # using a secondary graph to do things involving reservoir edges

		self.sw = 10000

		self.wedge_res = [None for i in range(self.sw)] # list to store reservoir sample of wedges


		self.is_closed = [False for i in range(self.sw)] # list to store that a wedge has been detected to be closed

		self.tot_wedges = 0

		self.t = 1
		T = 0

		# randomize ordering of edges
		edges = list(self.sampled_G.edges)
		random.shuffle(edges)

		for et in edges:
			# a couple datasets had an issue where edges didn't have 2 elements, workaround by ignoring any such case
			if len(et) != 2:
				pass
			self.birthday_update(et)
			p = self.is_closed.count(True) / len(self.is_closed) # p is fraction of wedges that are detected to be closed
			kt = 3*p
			# T is the number of triangles
			T = self.tot_wedges * (p*(self.t**2)) / (self.se * (self.se - 1))

			self.t += 1

			if self.t % 500 == 0:
				print(self.t) # trial
				print(T)      # triangle cnt
				print(kt)     #
				print(p)      # fraction of closed wedges in the reservoir
				print(self.tot_wedges) # total # of wedges
				print("----")
		return T

	def birthday_update(self, next_edge):
		"""
		part of birthday paradox alg
		"""
		# check if the new edge closes any wedge in the reservoir and update is_closed
		next_edge_as_set = set(next_edge)
		for	i in range(self.sw):
			if self.wedge_res[i] == None:  # some slot might not be filled
				continue
			if self.wedge_res[i] == next_edge_as_set:
				self.is_closed[i] = True

		updated = False
		removed_edges = []

		x = random.random()
		# success prob. of storing new edge in reservoir
		if x <= (1 - (1 - 1/self.t)**self.se):
			# choose random index in reservoir list to replace
			i = random.randint(0, self.se-1)
			if self.edge_res[i] is not None:
				removed_edges.append(self.edge_res[i])
			self.edge_res[i] = next_edge_as_set
			updated = True

		if updated: # if any update to edge_res
			# update total wedges and get number of wedges involving et
			N_t = self.birthday_update_tot_wedges(next_edge, removed_edges) # tot_wedge updated
			self.new_wedges = len(N_t)

			if self.tot_wedges > 0: # change some of the wedge reservoir entries with N_t
				for i in range(self.sw):
					x = random.random()
					if x <= self.new_wedges / self.tot_wedges:
						w = random.choice(N_t)
						self.wedge_res[i] = w
						self.is_closed[i] = False

	def birthday_update_tot_wedges(self, next_edge, removed_edges):
		"""
		part of birthday paradox alg.
		:return:
		"""
		next_edge_as_set = set(next_edge)

		# fix the total wedges count
		already_removed = []
		for edge in removed_edges:
			if edge not in already_removed and edge not in self.edge_res:
				edge_as_list = list(edge)
				# somehow, an edge with only 1 node sometimes gets to this point when running hep-th.
				# avoid error using try/except
				if len(edge_as_list) == 2:
					already_removed.append(edge)
					self.tot_wedges -= (self.BirthdayGraph.degree[edge_as_list[0]] - 1) + \
									   (self.BirthdayGraph.degree[edge_as_list[1]] - 1)

					try:
						self.BirthdayGraph.remove_edge(tuple(edge_as_list))
					except: # edge_res has duplicates; some edges are removed in the previous rounds
						pass
					for i in range(self.sw):
						if self.wedge_res[i] == edge:
							self.is_closed[i] = False

		self.BirthdayGraph.add_edge(next_edge[0], next_edge[1])
		# update total wedges with new wedges formed by adding this edge
		self.tot_wedges += (self.BirthdayGraph.degree[next_edge[0]] - 1) + \
						   (self.BirthdayGraph.degree[next_edge[1]] - 1)

		# get all new wedges made by the newly inserted edge

		# new wedges will be stored as the edge needed to close the underlying wedge
		new_wedges_with_et = [{next_edge[1], n1}
							  for n1 in self.BirthdayGraph.neighbors(next_edge[0]) if n1 != next_edge[1]] + \
							[{next_edge[0], n2}
							 for n2 in self.BirthdayGraph.neighbors(next_edge[1]) if n2 != next_edge[0]]

		return new_wedges_with_et

	def eigen_est(self):
		'''
		http://www.math.cmu.edu/~ctsourak/asonam_book.pdf

		:param
		'''
		tol = 0.0001
		p = 1

		A = scipy.sparse.lil_matrix(nx.adjacency_matrix(self.sampled_G), dtype=float)

		# stage 1: Achlioptas-McSherry Sparsification
		# similar to DOULIN, but retain all the edges
		assert(A.shape[0] == A.shape[1])

		n_cnt = A.shape[0]
		for i in range(n_cnt):
			for j in range(n_cnt):
				# toss a coin with success prob. p
				if random.random() < p:
					A[i, j] /= p
				else:
					pass

		# stage 2: EigenTriangle
		# initialization: i, lambda_vec

		# Lanczos  method: efficient  for  finding  the  top eigenvalues in sparse, symmetric matrices.

		lambda_vec_all, _ =  scipy.sparse.linalg.eigsh(A.toarray(), k = int(n_cnt * 0.5))
		# directly use the scipy implementation,
		# top x eigenvalues - for facebook p = 0.3, x = 0.3 is not enough
		def LanczosMethod(A, i):
			return lambda_vec_all[i-1]


		lambda_i = LanczosMethod(A, 1) # i = 1
		lambda_vec = [lambda_i]

		err = np.inf
		i = 2
		while err > tol:
			lambda_i = LanczosMethod(A, i)
			lambda_vec.append(lambda_i)
			i += 1

			err = np.abs(np.power(lambda_i, 3)) / np.sum(np.power(np.abs(lambda_vec), 3))

		cnt = np.sum(np.power(np.abs(lambda_vec[:-1]), 3)) / 6

		return cnt


	def trace_est(self):
		gamma = 3
		# For collaboration/citations graphs γ = 1 − 2
		# seems adequate, for social networks γ = 3 and for large
		# web graphs/communications networks γ = 4.
		# - section 6.2 (https://pdfs.semanticscholar.org/2471/6ee2bf34934e8eb70a7aca4ffa38b544ca81.pdf)
		adjacency_matrix = nx.to_numpy_matrix(self.sampled_G)
		n = len(adjacency_matrix)
		M = math.ceil(gamma * (np.log(n) ** 2))
		T = np.empty(M)
		for i in range(M):
			x = np.random.rand(n)
			y = np.matmul(adjacency_matrix,x)
			Ti = np.matmul(np.matmul(y,adjacency_matrix),y.transpose())
			T[i] = float(Ti)/6
		return np.sum(T)/M

	def trace_exact(self):
		a_sparse = nx.to_scipy_sparse_matrix(self.sampled_G)
		a3_sparse = a_sparse* a_sparse * a_sparse
		a3 = a3_sparse.todense()
		#a3 = matrix_power(nx.to_numpy_matrix(self.sampled_G),3)
		return float(a3.trace())/6


def analysis(res, ground_truth, save_file, alg_name, data_name):
	'''
	v.1 visualization of results

	:param res: [(p, cnt, running_time), ...]
	:ground_truth: (cnt, running_time)
	'''

	fig, axes = plt.subplots(1, 2, figsize = (10, 5))
	fig.suptitle("{} - {}".format(data_name, alg_name))

	axes[0].plot([p for (p, cnt, t) in res], [cnt for (p, cnt, t) in res], '.-', color = 'k', label="simulation")
	axes[0].plot([p for (p, cnt, t) in res], [ground_truth[0] * (p ** 3) for (p, cnt, t) in res], '.-', color = 'gray', label="estimate")
	#
	axes[0].set_title("Simulated vs Estimated triangle cnt. ")
	axes[0].set_xlabel("p")
	axes[0].set_ylabel("cnt.")

	axes[1].plot([p for (p, cnt, t) in res], [t for (p, cnt, t) in res], '.-', color = 'k', label = "simulation")
	axes[1].plot([p for (p, cnt, t) in res], [ground_truth[1] * (p ** 2) for (p, cnt, t) in res], '.-', color = 'gray', label= "estimate")
	#
	axes[1].set_title("Simulated vs Estimated run time ")
	axes[1].set_xlabel("p")
	axes[1].set_ylabel("running time (sec.)")

	for i in range(2):
		axes[i].legend()

	fig.savefig(save_file)

	return


def load_hepth():
	"""
	Load HEP-th dataset. other MATLAB matrix data can be loaded similarly
	:return:
	"""
	raw_dat = loadmat('data/HEP-th-new.mat')
	mat_dat = scipy.sparse.csr_matrix(raw_dat['Problem'][0][0][2])
	return nx.from_scipy_sparse_matrix(mat_dat)


def run_experiments(G, p_l, algs, trials, dataset_name):
	"""
	Run experiment on a loaded graph for multiple algorithms and p values.

	:param G: graph to count triangles for
	:param p_l: list of probabilities to run experiments for
	:param algs: list of strings used to choose the algorithm to run
	:param trials: number of trials to repeat the algorithm for
	:param dataset_name: name of the dataset, used for saving results
	"""
	result_dir = 'results/'+dataset_name
	os.mkdir(result_dir)
	for alg in algs:
		os.mkdir(result_dir+"/"+alg)
		res = [('trial', 'p', 'count', 'time')]
		for p in p_l:
			for trial in range(trials):
				print("trial:{}, p:{}".format(trial, p))
				counter = DOULION(G, p)
				t = time.time()
				cnt = counter.run(alg)
				print(cnt)
				run_time = time.time() - t
				res.append((trial, p, cnt, run_time))
				print(run_time)
		with open(result_dir+"/"+alg+'/result.txt', 'w') as f:
			for item in res:
				f.write("{}\n".format(item))


if __name__ == "__main__":

	# load dataset.
	# G = nx.read_edgelist("facebook_combined.txt", delimiter = ' ', data = (('w', int),))
	G = load_hepth()

	# p values and algorithm choices to run experiment
	p_l = [0.1, 0.3, 0.5, 0.7, 1]
	algs = ['node_iter', 'edge_iter', 'trace_exact']

	# run experiment. currently set up to run on HEP-th dataset
	run_experiments(G, p_l, algs, trials=5, dataset_name='hep_th')

