import networkx as nx
import numpy as np 
import random
import matplotlib.pyplot as plt 
import time
from numpy.random import choice
import math

class WEDGE:
    def __init__(self,G):
        self.G=G
        self.d=dict(G.degree)
        return
    def Wedge_v(self): # all wedges for each vertices in the G
        Wv=dict()
        for i in self.d.keys():
            if self.d[i]>=2:
                Wv[i]=math.factorial(self.d[i])/(2*math.factorial(self.d[i]-2))
        return Wv
    def P_v(self, Wv_dict): # all probabilities, and the total wedge number in graph G
        Pv=dict()
        W=sum(Wv_dict.values())
        for i in Wv_dict.keys():
            Pv[i]=Wv_dict[i]/W
        return Pv, W
    def Random_vertices(self, prob_dict, k):
        K_vertices=choice(list(prob_dict.keys()),k, p=list(prob_dict.values()),replace=False) #remove duplication
        return K_vertices # a list
    def Random_neighbors(self, candidate_vertices):
        Rs=dict()
        for i in candidate_vertices:
            Rs[i]=random.sample(population=list(self.G.neighbors(i)),k=2)
        return Rs
    def Count_closed_wedges(self,Rs,selected_vertices_k):
        t=0
        used_keys=[]
        for i in Rs.keys():
            key1=Rs[i][0]
            key2=Rs[i][1]
            if ((key2 in Rs.keys()) and (key1 in Rs[key2])) \
            or ((key1 in Rs.keys()) and (key2 in Rs[key1])):
                t+=1
            else:
                t=t
            used_keys.append(i)
        K=3*t/selected_vertices_k
        return K #a fraction
    def triangle_count(self, K, total_wedges):
        Triangle_number=1/3*K*total_wedges
        return Triangle_number
if __name__ == "__main__":
    G=nx.Graph()
    e=nx.read_edgelist("facebook_combined.txt")
    G.add_edges_from(e.edges())
    G.add_nodes_from(e.nodes())
    d=dict(G.degree)
    counter=WEDGE(G)
    Wv=counter.Wedge_v() #Wedge number for each vertice
    print (Wv)
    Pv,T_wedge=counter.P_v(Wv) #Probability dictionary & total wedges
    print(T_wedge)
    Choose_vertice=3900
    Rv=counter.Random_vertices(Pv,Choose_vertice) #Random vertices
    Rn=counter.Random_neighbors(Rv) #Random neighbors\
    Cw=counter.Count_closed_wedges(Rn,Choose_vertice) #closed wedges
    print(Cw)
    Tn=counter.triangle_count(Cw, T_wedge) #triangle number
    print(Tn)
    