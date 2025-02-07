import matplotlib.pyplot as plt
import pickle

type_num = 3

with open(f'grid_world_2_data/Values/Correct_thetaList_1', "rb") as pkl_rb_obj:
    theta_list = pickle.load(pkl_rb_obj)

with open(f'grid_world_2_data/Values/Correct_entropy_1', "rb") as pkl_rb_obj:
    entropy_list = pickle.load(pkl_rb_obj)

plt.rcParams.update({'font.size': 14})

iter_num = 2000
iteration_list = range(iter_num)
plt.plot(iteration_list, entropy_list, label=r"conditional entropy $H(S_0|Y;\theta)$")
plt.xlabel("The iteration number")
plt.ylabel(r"conditional entropy")
plt.title("The Conditional Entropy for Each Iteration")
plt.legend()
# plt.savefig(f'./grid_world_2_data/Graphs/CorrectEx_{ex_num}_dynaNoi01_obsNoi01.png')
plt.savefig(f'./grid_world_2_data/Graphs/CorrectEx_1_dynaNoi01_obsNoi01.pdf', format="pdf", bbox_inches="tight")
plt.show()

with open(f'./grid_world_2_data/Values/Results_Analysis_Fixed/type_1_list_r_{type_num}', "rb") as pkl_rb_obj:
    type_1_list_r = pickle.load(pkl_rb_obj)
print(type_1_list_r[-1])

with open(f'./grid_world_2_data/Values/Results_Analysis_Fixed/type_2_list_r_{type_num}', "rb") as pkl_rb_obj:
    type_2_list_r = pickle.load(pkl_rb_obj)
print(type_2_list_r[-1])

with open(f'./grid_world_2_data/Values/Results_Analysis_Fixed/type_3_list_r_{type_num}', "rb") as pkl_rb_obj:
    type_3_list_r = pickle.load(pkl_rb_obj)
print(type_3_list_r[-1])

with open(f'./grid_world_2_data/Values/Results_Analysis_Fixed/type_1_list_{type_num}', "rb") as pkl_rb_obj:
    type_1_list = pickle.load(pkl_rb_obj)
print(type_1_list[-1])

with open(f'./grid_world_2_data/Values/Results_Analysis_Fixed/type_2_list_{type_num}', "rb") as pkl_rb_obj:
    type_2_list = pickle.load(pkl_rb_obj)
print(type_2_list[-1])

with open(f'./grid_world_2_data/Values/Results_Analysis_Fixed/type_3_list_{type_num}', "rb") as pkl_rb_obj:
    type_3_list = pickle.load(pkl_rb_obj)
print(type_3_list[-1])

T = 10
# Create plot
fig, ax = plt.subplots()
# Plot lines
line1, = ax.plot(range(T), type_1_list_r, ':b.', label='type 1 (random)')
line2, = ax.plot(range(T), type_2_list_r, ':rD', label='type 2 (random)')
line3, = ax.plot(range(T), type_3_list_r, ':gs', label='type 3 (random)')
line4, = ax.plot(range(T), type_1_list, '-b.', label='type 1 (min_entropy)')
line5, = ax.plot(range(T), type_2_list, '-rD', label='type 2 (min_entropy)')
line6, = ax.plot(range(T), type_3_list, '-gs', label='type 3 (min_entropy)')
plt.xlabel(r"The time step $t$")
plt.ylabel("The belief (posterior initial distribution)")
plt.title(f"The Evolution of Belief (True Type is {type_num})")
# Create first legend for the first 3 lines
first_legend = ax.legend([line1, line2, line3],
                         ['type 1 (rand)','type 2 (rand)','type 3 (rand)'],
                         loc='center left', bbox_to_anchor=(0.55, 0.63))
# Add the first legend manually to the plot
ax.add_artist(first_legend)
# Create a second legend for other lines
ax.legend([line4, line5, line6],
          ['type 1 (ours)', 'type 2 (ours)', 'type 3 (ours)'],
          loc='center left', bbox_to_anchor=(0.55, 0.37))
plt.savefig(f'./grid_world_2_data/Graphs/Results_Analysis_Fixed/type_{type_num}.png')
plt.show()