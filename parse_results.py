import numpy as np
from linear_py import analysis
import matplotlib.pyplot as plt


def analysis_with_errorbars(res, ground_truth, save_file, alg_name, data_name):
	'''
	v.1

	:param res: [(p, cnt, running_time), ...]
	:ground_truth: (cnt, running_time)
	'''
	fig, axes = plt.subplots(1, 2, figsize = (10, 5))
	fig.suptitle("{} - {}".format(data_name, alg_name))

	axes[0].plot([p for (p, cnt, cnt_std, t, t_std) in res], [cnt for (p, cnt, cnt_std, t, t_std) in res], '.-', color = 'k')
	axes[0].errorbar([p for (p, cnt, cnt_std, t, t_std) in res], [cnt for (p, cnt, cnt_std, t, t_std) in res], yerr=[cnt_std for (p, cnt, cnt_std, t, t_std) in res], linestyle='none')
	axes[0].axhline(y=ground_truth[0], linestyle='dotted', color='r')
	#
	axes[0].set_title("Doulin Triangle cnt. ")
	axes[0].set_xlabel("p")
	axes[0].set_ylabel("cnt.")

	axes[1].plot([p for (p, cnt, cnt_std, t, t_std) in res], [t for (p, cnt, cnt_std, t, t_std) in res], '.-', color = 'k')
	axes[1].errorbar([p for (p, cnt, cnt_std, t, t_std) in res], [t for (p, cnt, cnt_std, t, t_std) in res], yerr=[t_std for (p, cnt, cnt_std, t, t_std) in res], linestyle='none')
	axes[1].axhline(y=ground_truth[1], linestyle='dotted', color='r')
	#
	axes[1].set_title("Doulin Runtime")
	axes[1].set_xlabel("p")
	axes[1].set_ylabel("running time (sec.)")

	fig.savefig(save_file)

def accuracy_speedup_dots(res, ground_truth, save_file, alg_name, data_name):
	'''
	v.1

	:param res: [(p, cnt, running_time), ...]
	:ground_truth: (cnt, running_time)
	'''
	res_accuracy = [1-np.abs((cnt/(p**3)/ground_truth[0] - 1)) for (p, cnt, t) in res]
	res_speedup = [ground_truth[1]/t for (p, cnt, t) in res]

	plt.rcParams.update({'font.size': 13})
	fig, ax = plt.subplots(1, 1)
	fig.suptitle("{} - {}".format(data_name, alg_name))
	ax.plot(res_accuracy, res_speedup, 'ro')
	p = [0.1, 0.3, 0.5, 0.7, 1.0]
	for i in range(5):
		print(res_accuracy[i])
		ax.annotate(p[i], (res_accuracy[i], res_speedup[i]))

	ax.set_xlabel("Accuracy")
	ax.set_ylabel("Speedup")
	plt.locator_params(axis='x', nbins=4)
	# plt.xticks([0.8, 0.9, 0.95, 1.0])
	plt.locator_params(axis='y', nbins=7)
	# ax.set_xlim([0.98, 1.0])
	#ax.set_ylim([0, 120])


	fig.savefig(save_file)


def main():
	vis_sets = [
		('Trace-Exact', 'Epinions', 'results/epinions/', 'trace_exact/'),
		# ('Edge Iterator', 'EAT_RS', 'results/EAT_RS/', 'edge_iter/'),
		# ('Trace-Exact', 'EAT_RS', 'results/EAT_RS/', 'trace_exact/'),
	]
	for (alg_name, data_name, dataset, alg) in vis_sets:
		file = dataset + alg + 'result.txt'
		# savefile = dataset+alg+"result.png"

		lines = []
		with open(file, 'r') as f:
			for l in f:
				lines.append(l)

		lines = lines[1:]
		for i in range(len(lines)):
			lines[i] = lines[i][1:-2]

		for i in range(len(lines)):
			lines[i] = [float(s) for s in lines[i].split(",")]

		p_count_dict = {0.1:[], 0.3:[], 0.5:[], 0.7:[], 1.0:[]}
		p_time_dict = {0.1:[], 0.3:[], 0.5:[], 0.7:[], 1.0:[]}

		for l in lines:
			p_count_dict[l[1]].append(l[2])
			p_time_dict[l[1]].append(l[3])

		res = []
		for p in [0.1, 0.3, 0.5, 0.7, 1.0]:
			res.append((p, np.mean(p_count_dict[p]), np.mean(p_time_dict[p]))) #
		accuracy_speedup_dots(res, (res[-1][1], res[-1][2]), dataset+alg+"acc_speed_res.png", alg_name, data_name)
		analysis(res, (res[-1][1], res[-1][2]), dataset+alg+"analysis_est_vs_sim.png", alg_name, data_name)

		# For if we want error bars / something with std
		res=[]
		for p in [0.1, 0.3, 0.5, 0.7, 1.0]:
			res.append((p, np.mean(p_count_dict[p])/(p**3), np.std(p_count_dict[p])/(p**3), np.mean(p_time_dict[p])/(p**2), np.std(p_time_dict[p])/(p**2)))

		analysis_with_errorbars(res, (res[-1][1], res[-1][3]), dataset+alg+"result_with_errorbars.png", alg_name, data_name)

if __name__ == '__main__':
	main()
