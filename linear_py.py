from collections import defaultdict

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
		
		# algorithm 1: Streaming-Triangles (Se, Sw)
		
		# wedges are tuples of two edges
		# a wedge is a path of length 2.

		print(len(self.sampled_G.edges))
		

		self.se = int(0.1 * len(self.sampled_G.edges)) #parameter that tells us how many edges to store.
		print(self.se)
		# self.edge_res = [ None for i in range(self.se)] # list to store reservoir sample of edges
		self.edge_res = random.sample(self.sampled_G.edges, self.se)
		self.BirthdayGraph = nx.Graph() # using a secondary graph to do things involving reservoir edges
		self.BirthdayGraph.add_edges_from(self.edge_res)

		self.sw = int(0.01 * (self.se * max(self.sampled_G.degree(), key = lambda t:t[1])[1]))  #parameter that tells us how many wedges to store.
		print(self.sw)
		# experiments in the paper set se and sw to 10K, probably need to find a good number to use
		# se and sw should be based on sample_G
		
		self.wedge_res = [None for i in range(self.sw)] # list to store reservoir sample of wedges
		# maintains a uniform sample of the wedges created by the edge reservoir at any step of the process. 
		# (The wedge reservoir may include wedges whose edges are no longer in the edge reservoir.)


		self.is_closed = [False for i in range(self.sw)] # list to store that a wedge has been detected to be closed
							# TODO: create this vector based on self.wedge_res
		for i in range(self.sw):
			found = False
			while not found:
				try:
					n1 = random.choice(list(self.BirthdayGraph.nodes))
					mid_point = random.choice(list(self.BirthdayGraph[n1].keys()))
					n2 = random.choice(list(self.BirthdayGraph[mid_point].keys()))
					if n1 != n2:
						found = True
				except IndexError:
					continue

			self.wedge_res[i] = ((n1, mid_point), (n2, mid_point))

			if self.BirthdayGraph.has_edge(n1, n2):
				self.is_closed[i] = True

		self.tot_wedges = 0
		for i in range(self.se - 1):
			for j in range(i+1, self.se):
				if len(set(self.edge_res[i]).intersection(self.edge_res[j])) != 0:
					self.tot_wedges += 1


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
			#  getting triangle count of 8,000,000 instead of expected 1,600,000
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

	def birthday_update(self, et):
		"""
		part of birthday paradox alg
		:param et:
		:return:
		"""
		for	i in range(self.sw):
			if self.wedge_res[i] == None:  # some slot might not be filled
				continue
			# e_t closes the wedge in wedge_res i
			# print(self.wedge_res[i][0], self.wedge_res[i][1], et)
			if len(set(self.wedge_res[i][0]).intersection(et)) == 1 and \
					len(set(self.wedge_res[i][1]).intersection(et)) == 1:
				self.is_closed[i] = True  #

		updated = False
		removed_edges = []
		for i in range(self.se):
			x = random.random()
			if x <= 1/self.t: # 
				if self.edge_res[i] is not None: # 
					removed_edges.append(tuple(self.edge_res[i])) # some might be duplicated
				self.edge_res[i] = et
				updated = True
		removed_edges = list(set(removed_edges))

		if updated: # if any update to edge_res
			# update tot wedges and get number of wedges involving et
			N_t = self.birthday_update_tot_wedges(et, removed_edges) # tot_wedge updated
			self.new_wedges = len(N_t)

			if self.tot_wedges > 0: # change some of the wedge reservoir entries with N_t
				for i in range(self.sw):
					x = random.random()
					if x <= self.new_wedges / self.tot_wedges:
						w = random.choice(N_t)
						self.wedge_res[i] = w
						self.is_closed[i] = False   # didn't check; (?)

	def birthday_update_tot_wedges(self, et, removed_edges):
		"""
		part of birthday paradox alg.
		:return:
		"""

		# fix the total wedges count
		# print("here!")
		for edge in removed_edges:
			self.tot_wedges -= (len(self.BirthdayGraph.edges(list(edge)[0])) - 1) + \
							   (len(self.BirthdayGraph.edges(list(edge)[1])) - 1)

			try:
				self.BirthdayGraph.remove_edge(list(edge)[0], list(edge)[1]) 
			except: # edge_res has duplicates; some edges are removed in the previous rounds
				pass

		self.BirthdayGraph.add_edge(list(et)[0], list(et)[1])
		# update total wedges with new wedges formed by adding this edge
		self.tot_wedges += (len(self.BirthdayGraph.edges(list(et)[0])) - 1) + \
						   (len(self.BirthdayGraph.edges(list(et)[1])) - 1)

		# get all new wedges made by the newly inserted edge

		new_wedges_with_et = [((list(et)[0], n1), et) 
								for n1 in nx.neighbors(self.BirthdayGraph, list(et)[0]) if n1 != list(et)[1]] + \
							 [((list(et)[1], n2), et)
							 	for n2 in nx.neighbors(self.BirthdayGraph, list(et)[1]) if n2 != list(et)[0]]

		return new_wedges_with_et


def analysis(res, ground_truth):
	'''
	v.1

	:param res: [(p, cnt, running_time), ...]
	:ground_truth: (cnt, running_time)
	'''



	fig, axes = plt.subplots(1, 2, figsize = (10, 5))

	axes[0].plot([p for (p, cnt, t) in res], [(cnt / (p**3)) for (p, cnt, t) in res], '.-', color = 'k', label="simulation")
	axes[0].plot([p for (p, cnt, t) in res], [ground_truth[0] for (p, cnt, t) in res], '.-', color = 'gray', label="estimate")
	axes[0].set_title("Simulated vs Estimated triangle cnt. ")
	axes[0].set_xlabel("p")
	axes[0].set_ylabel("cnt.")

	axes[1].plot([p for (p, cnt, t) in res], [t for (p, cnt, t) in res], '.-', color = 'k', label = "simulation")
	axes[1].plot([p for (p, cnt, t) in res], [ground_truth[1] for (p, cnt, t) in res], '.-', color = 'gray', label= "estimate")
	axes[1].set_title("Simulated vs Estimated running_time ")
	axes[1].set_xlabel("p")
	axes[1].set_ylabel("running time (sec.)")

	for i in range(2):
		axes[i].legend()

	fig.savefig("./log/result_image.png")

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


