from collections import defaultdict

import networkx as nx
import numpy as np
import random
import matplotlib.pyplot as plt
import time


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
		elif alg == "birthday":
			cnt = self.birthday_paradox()
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

	def birthday_paradox(self):
		"""
		implemented using http://delivery.acm.org/10.1145/2710000/2700395/a15-jha.pdf?ip=129.161.86.178&id=2700395&acc=ACTIVE%20SERVICE&key=7777116298C9657D%2EAF047EA360787914%2E4D4702B0C3E38B35%2E4D4702B0C3E38B35&__acm__=1573168257_e346bde5ea41699f6b4fc6f62572ca2a

		:return:
		"""
		#wedges are tuples of two edges
		print(len(self.G.edges))
		self.BirthdayGraph = nx.Graph() # using a secondary graph to do things involving reservoir edges

		self.se = 5000 #parameter that tells us how many edges to store.
		self.sw = 5000 #parameter that tells us how many wedges to store.
		# experiments in the paper set se and sw to 10K, probably need to find a good number to use
		self.edge_res = [None for i in range(self.se)] # list to store reservoir sample of edges
		self.wedge_res = [None for i in range(self.sw)] # list to store reservoir sample of wedges
		self.is_closed = [False for i in range(self.sw)] # list to store that a wedge has been detected to be closed
		self.tot_wedges = 0
		self.t = 1
		T = 0

		# since this is an algorithm for streaming, maybe we should make edges we get from G be randomized.
		edges = list(self.G.edges)
		random.shuffle(edges)
		for et in edges:
			et = set(et) # convert edge represented sa tuple into set

			self.birthday_update(et)
			p = self.is_closed.count(True) / len(self.is_closed) # p is fraction of wedges that are detected to be closed
			kt = 3*p
			#TODO: something is wrong here. T keeps increasing because of the term self.t**2.
			#      probably missing an extra step in the paper, or misinterpreting what t in algorithm 1 is
			T = self.tot_wedges * (p*(self.t)**2) / (self.se * (self.se - 1)) # T is the number of triangles

			self.t += 1

			if self.t % 500 == 0:
				print(self.t)
				print(T)
		return T

	def birthday_update(self, et):
		"""
		part of birthday paradox alg
		:param et:
		:return:
		"""
		for	i in range(self.sw):
			if self.wedge_res[i] == None:
				continue
			#et closes the wedge in wedge_res i
			if len(self.wedge_res[i][0].intersection(et)) == 1 and \
					len(self.wedge_res[i][1].intersection(et)) == 1:
				self.is_closed[i] = True

		updated = False
		for i in range(self.se):
			x = random.random()
			if x <= 1/self.t:
				if self.edge_res[i] is not None:
					to_rem = list(self.edge_res[i])
					try: #exception raised if edge isn't in graph.
						# update total wedges to reflect losing this edge
						self.tot_wedges -= (len(self.BirthdayGraph.edges(to_rem[0])) - 1) +\
										   (len(self.BirthdayGraph.edges(to_rem[1])) - 1)
						self.BirthdayGraph.remove_edge(to_rem[0], to_rem[1])
					except:
						pass
				self.edge_res[i] = et
				list_et = list(et)
				self.BirthdayGraph.add_edge(list_et[0], list_et[1])
				# update total wedges with new wedges formed by adding this edge
				self.tot_wedges += (len(self.BirthdayGraph.edges(list_et[0])) - 1) +\
								   (len(self.BirthdayGraph.edges(list_et[1])) - 1)
				updated = True

		if updated:
			#update tot wedges and get number of wedges involving et
			self.N = self.birthday_update_tot_wedges(et)
			self.new_wedges = len(self.N)
			if self.tot_wedges > 0:
				for i in range(self.sw):
					x = random.random()
					if x <= self.new_wedges / self.tot_wedges:
						w = random.choice(self.N)
						self.wedge_res[i] = w
						self.is_closed[i] = False

	def birthday_update_tot_wedges(self, et):
		"""
		part of birthday paradox alg.
		:return:
		"""
		list_et = list(et)
		new_wedges_with_et = []
		for edge in self.BirthdayGraph.edges(list_et[0]):
			if set(edge) != et:
				new_wedges_with_et.append((set(edge), et))

		for edge in self.BirthdayGraph.edges(list_et[1]):
			if set(edge) != et:
				new_wedges_with_et.append((set(edge), et))
		return new_wedges_with_et


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
	G = nx.read_edgelist("facebook_combined.txt", delimiter = ' ', data = (('w', int),))
	p_l = [0.1, 0.3, 0.5, 0.7, 1]
	res = []

	for p in p_l:
		counter = DOULION(G, p)
		t = time.time()
		cnt = counter.run("birthday")#("node_iter")
		run_time = time.time() - t
		res.append((p, cnt, run_time))
		print("p: ", p, ", triangle cnt: ", cnt)

	analysis(res, (res[-1][1], res[-1][2]))

	
