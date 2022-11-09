import os
import pandas as pd

# ini variables
t_ids_list = [198664]
cur_case = 'slurp' # slurp # slue_voxpopuli
t_csv_file_name = 'selected_aux_train_wi_syn_label.csv'
t_attr = 'ori_source'
root_folder_path = '/home/jfhe/Documents/MountHe/jfhe/projects/speechbrain/recipes/SLURP/Zero_shot_cross_modal/results'

if cur_case == 'slue_voxpopuli':
    sec_folder_path = 'slue-voxpopuli/slue-voxpopuli_slue-voxpopuli-full/'
    t_source = ['slue-voxpopuli', 'slue-voxpopuli-full']
elif cur_case == 'slurp':
    sec_folder_path = 'slurp/slurp_peoplespeech/'
    t_source = ['slurp', 'peoplespeech']

for i in range(len(t_ids_list)):
    fir_folder_path = str(t_ids_list[i])
    total_folder_path = os.path.join(root_folder_path, fir_folder_path, sec_folder_path)
    final_file_path = os.path.join(total_folder_path, t_csv_file_name)
    source_df = pd.read_csv(final_file_path, header=0, sep=',')
    res_list = []
    for j in range(len(t_source)):
        mid_data = source_df[(source_df[t_attr]==t_source[j])]
        res_list.append(mid_data.shape[0])


        print(f'{t_ids_list[i]} has {t_source[j]} as {res_list[j]}')
    print('\n')