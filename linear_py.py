import networkx as nx
import numpy as np
import random
import matplotlib.pyplot as plt
import time
import math
from numpy.linalg import matrix_power


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
			if random.random() < p:
				# G.edges[e]['w'] = 1/p
				self.sampled_G.add_edge(e[0], e[1])
			else:
				pass

		triangle_cnt = self.tri_count(triangle_cnt_alg)

		return triangle_cnt

	def tri_count(self, alg):
		'''
		:param alg: 
			"node_iter", 
			"edge_iter".
			"trace_est",
			"wedge_sampling"
		'''

		if alg == "node_iter":
			cnt = self.node_iter()
		elif alg == "trace_est":
			cnt = self.trace_est()
		elif alg == "trace_exact":
			cnt = self.trace_exact()
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
	axes[0].plot([p for (p, cnt, t) in res], [ground_truth[0] * (p**3) for (p, cnt, t) in res], '.-', color = 'gray', label="estimate")
	axes[0].set_title("Simulated vs Estimated triangle cnt. ")
	axes[0].set_xlabel("p")
	axes[0].set_ylabel("cnt.")

	axes[1].plot([p for (p, cnt, t) in res], [t for (p, cnt, t) in res], '.-', color = 'k', label = "simulation")
	axes[1].plot([p for (p, cnt, t) in res], [ground_truth[1] * (p**2) for (p, cnt, t) in res], '.-', color = 'gray', label= "estimate")
	axes[1].set_title("Simulated vs Estimated running_time ")
	axes[1].set_xlabel("p")
	axes[1].set_ylabel("running time (sec.)")

	for i in range(2):
		axes[i].legend()

	fig.savefig("./log/node_iter_facebook.png")

	return



if __name__ == "__main__":

	# setting
	G = nx.read_edgelist("facebook_combined.txt")
	p_l = [0.1, 0.3, 0.5, 0.7, 1]
	res = []

	for p in p_l:
		counter = DOULION(G, p)
		t = time.time()
		cnt = counter.run("trace_est")
		run_time = time.time() - t
		res.append((p, cnt, run_time))
		print("p: ", p, ", triangle cnt: ", cnt)

	analysis(res, (res[-1][1], res[-1][2]))

	
