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
			"trace_est",
			"wedge_sampling"

		'''

		for e in G.edges():
			# toss a coin with success prob. p
			if random.random() < self.p:
				# G.edges[e]['w'] = 1/p
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
			nodeset_n = set([s for s in self.sampled_G.neighbors(n)])# if n < s and s < m])
			nodeset_m = set([s for s in self.sampled_G.neighbors(m)])# if n < s and s < m])
			num_triangles += len(nodeset_m.intersection(nodeset_n))

		return num_triangles / 3

	def birthday_paradox(self):
		"""
		implemented using http://delivery.acm.org/10.1145/2710000/2700395/a15-jha.pdf?ip=129.161.86.178&id=2700395&acc=ACTIVE%20SERVICE&key=7777116298C9657D%2EAF047EA360787914%2E4D4702B0C3E38B35%2E4D4702B0C3E38B35&__acm__=1573168257_e346bde5ea41699f6b4fc6f62572ca2a

		:return:
		"""

		# algorithm 1: Streaming-Triangles (Se, Sw)

		# wedges are tuples of two edges
		# a wedge is a path of length 2.

		print(len(self.sampled_G.edges))


		self.se = 10000
		print(self.se)
		self.edge_res = [ None for i in range(self.se)] # list to store reservoir sample of edges
		self.BirthdayGraph = nx.Graph() # using a secondary graph to do things involving reservoir edges

		self.sw = 10000
		print(self.sw)
		# experiments in the paper set se and sw to 10K, probably need to find a good number to use
		# se and sw should be based on sample_G

		self.wedge_res = [None for i in range(self.sw)] # list to store reservoir sample of wedges
		# maintains a uniform sample of the wedges created by the edge reservoir at any step of the process.
		# (The wedge reservoir may include wedges whose edges are no longer in the edge reservoir.)


		self.is_closed = [False for i in range(self.sw)] # list to store that a wedge has been detected to be closed

		self.tot_wedges = 0

		self.t = 1
		T = 0

		# since this is an algorithm for streaming, maybe we should make edges we get from G be randomized.
		edges = list(set(self.sampled_G.edges) - set(self.BirthdayGraph.edges))
		random.shuffle(edges)
		for et in edges:
			self.birthday_update(et)
			p = self.is_closed.count(True) / len(self.is_closed) # p is fraction of wedges that are detected to be closed
			kt = 3*p
			#TODO: something is probably wrong with the estimate
			#  possibly missing an extra step in the paper, or misinterpreting what t in algorithm 1 is
			T = self.tot_wedges * (p*(self.t**2)) / (self.se * (self.se - 1)) # T is the number of triangles

			self.t += 1  # ? trials?

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
		:param next_edge:
		:return:
		"""
		next_edge_as_set = set(next_edge)
		for	i in range(self.sw):
			if self.wedge_res[i] == None:  # some slot might not be filled
				continue
			if self.wedge_res[i] == next_edge_as_set:
				self.is_closed[i] = True  #

		updated = False
		removed_edges = []

		x = random.random()
		if x <= (1 - (1 - 1/self.t)**self.se):
		# for i in range(self.se):
		# 	if x <= 1/self.t: #
			i = random.randint(0, self.se-1)
			if self.edge_res[i] is not None: #
				removed_edges.append(self.edge_res[i]) # some might be duplicated
			self.edge_res[i] = next_edge_as_set
			updated = True
#				break

		if updated: # if any update to edge_res
			# update tot wedges and get number of wedges involving et
			N_t = self.birthday_update_tot_wedges(next_edge, removed_edges) # tot_wedge updated
			self.new_wedges = len(N_t)

			if self.tot_wedges > 0: # change some of the wedge reservoir entries with N_t
				for i in range(self.sw):
					x = random.random()
					if x <= self.new_wedges / self.tot_wedges:
						w = random.choice(N_t)
						self.wedge_res[i] = w
						self.is_closed[i] = False   # didn't check; (?)

	def birthday_update_tot_wedges(self, next_edge, removed_edges):
		"""
		part of birthday paradox alg.
		:return:
		"""
		next_edge_as_set = set(next_edge)

		# fix the total wedges count
		# print("here!")
		already_removed = []
		for edge in removed_edges:
			if edge not in already_removed and edge not in self.edge_res:
				already_removed.append(edge)
				edge_as_list = list(edge)
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
		#seems adequate, for social networks γ = 3 and for large
		#web graphs/communications networks γ = 4.
		# - section 6.2 (https://pdfs.semanticscholar.org/2471/6ee2bf34934e8eb70a7aca4ffa38b544ca81.pdf)
		adjacency_matrix = nx.to_numpy_matrix(self.sampled_G)
		n = len(adjacency_matrix)
		M = math.ceil(gamma * (np.log(n) ** 2))
		T = np.empty(M)
		for i in range(M):
			x = np.random.rand(n)
			y = np.matmul(x,adjacency_matrix)
			Ti = np.matmul(np.matmul(y,adjacency_matrix),y.transpose())
			T[i] = float(Ti)/6
		return np.sum(T)/M

	def trace_exact(self):
		a3 = matrix_power(nx.to_numpy_matrix(self.sampled_G),3)
		return float(a3.trace())/6


def analysis(res, ground_truth):
	'''
	v.1

	:param res: [(p, cnt, running_time), ...]
	:ground_truth: (cnt, running_time)
	'''



	fig, axes = plt.subplots(1, 2, figsize = (10, 5))

	axes[0].plot([p for (p, cnt, t) in res], [cnt for (p, cnt, t) in res], '.-', color = 'k', label="simulation")
	axes[0].plot([p for (p, cnt, t) in res], [ground_truth[0] * (p ** 3) for (p, cnt, t) in res], '.-', color = 'gray', label="estimate")
	#
	axes[0].set_title("Simulated vs Estimated triangle cnt. ")
	axes[0].set_xlabel("p")
	axes[0].set_ylabel("cnt.")

	axes[1].plot([p for (p, cnt, t) in res], [t for (p, cnt, t) in res], '.-', color = 'k', label = "simulation")
	axes[1].plot([p for (p, cnt, t) in res], [ground_truth[1] * (p ** 2) for (p, cnt, t) in res], '.-', color = 'gray', label= "estimate")
	#
	axes[1].set_title("Simulated vs Estimated running_time ")
	axes[1].set_xlabel("p")
	axes[1].set_ylabel("running time (sec.)")

	for i in range(2):
		axes[i].legend()

	fig.savefig("./log/log_birthday_newest_modifiedsesw.png")


	return

def load_hepth():
	raw_dat = loadmat('data/HEP-th-new.mat')
	mat_dat = scipy.sparse.csr_matrix(raw_dat['Problem'][0][0][2])
	return nx.from_scipy_sparse_matrix(mat_dat)

def run_experiments(G, p_l, algs, trials, dataset_name):
	result_dir = 'results/'+dataset_name
	os.mkdir(result_dir)
	for alg in algs:
		os.mkdir(result_dir+"/"+alg)
		res = [('trial', 'p', 'count', 'time')]
		for p in p_l:
			for trial in range(trials):
				counter = DOULION(G, p)
				t = time.time()
				cnt = counter.run("node_iter")  # ("node_iter")
				run_time = time.time() - t
				res.append((trial, p, cnt, run_time))
		with open(result_dir+"/"+alg+'/result.txt', 'w') as f:
			for item in res:
				f.write("{}\n".format(item))


if __name__ == "__main__":

	# setting
	G = nx.read_edgelist("facebook_combined.txt", delimiter = ' ', data = (('w', int),))
	p_l = [0.1, 0.3, 0.5, 0.7, 1]
	det_algs = ['node_iter', 'edge_iter', 'trace_exact']

	# following 2 lines for running experiments over hep-th-new
	#G = load_hepth()
	#run_experiments(G, p_l, det_algs, trials=5, dataset_name='hep_th')

	res = []
	for p in p_l:
		counter = DOULION(G, p)
		t = time.time()
		cnt = counter.run("node_iter")#("node_iter")
		run_time = time.time() - t
		res.append((p, cnt, run_time))
		print("p: ", p, ", triangle cnt: ", cnt)
		print("run time: ", run_time)

	analysis(res, (res[-1][1], res[-1][2]))

	# TODO: clean up to make running experiments straightforward
	# note: use fast_gnp_random_graph for testing on ER graph

