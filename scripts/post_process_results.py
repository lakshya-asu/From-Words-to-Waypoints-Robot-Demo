import pickle, json
import numpy as np
from hydra_python.utils import load_eqa_data
from omegaconf import OmegaConf

if __name__ == "__main__":
    cfg_path = "../cfg/grapheqa_habitat.yaml"
    cfg = OmegaConf.load(cfg_path)
    OmegaConf.resolve(cfg)

    questions_data, init_pose_data = load_eqa_data(cfg.data)

    filepath = '../outputs/' + cfg.exp_name
    outfile = filepath + '/metrics_new_succ.json'
    outfile2 = filepath + '/task_categories.json'

    #with open(filepath + '/' + cfg.exp_name + '_images_True.json', 'r') as file:
    with open (filepath + '/' + cfg.results_filename + '.json', 'r') as file:
        data = json.load(file)

    num_success, planning_steps_all_trajs, length_all_trajs = 0, 0, 0
    total_trajs = len(data.keys())

    identification, existence, count, state, location = 0, 0, 0, 0, 0
    identification_succ, existence_succ, count_succ, state_succ, location_succ = 0, 0, 0, 0, 0
    for question_ind, question_data in enumerate(questions_data):
        experiment_id = f'{question_ind}_{question_data["scene"]}_{question_data["floor"]}'
        if experiment_id in data.keys():
            if question_data['label'] == 'identification':
                identification += 1
                if data[experiment_id]['Success']:
                    identification_succ += 1

            elif question_data['label'] == 'existence':
                existence+=1
                if data[experiment_id]['Success']:
                    existence_succ += 1
            elif question_data['label'] == 'count':
                count+=1
                if data[experiment_id]['Success']:
                    count_succ += 1
            elif question_data['label'] == 'state':
                state+=1
                if data[experiment_id]['Success']:
                    state_succ += 1
            elif question_data['label'] == 'location':
                location+=1
                if data[experiment_id]['Success']:
                    location_succ += 1
            else:
                raise NotImplementedError("invalid question type")


    type_results = {}
    type_results['identification'] = identification
    type_results['existence'] = existence
    type_results['count'] = count
    type_results['state'] = state
    type_results['location'] = location
    type_results['total_trajs'] = total_trajs

    type_results['identification_succ'] = identification_succ/identification*100
    type_results['existence_succ'] = existence_succ/existence*100
    type_results['count_succ'] = count_succ/count*100
    type_results['state_succ'] = state_succ/state*100
    type_results['location_succ'] = location_succ/location*100

    print(f"Saving file: {outfile2}")
    with open(outfile2, 'w') as file:
        json.dump(type_results, file, indent=4)
    print(f"Saved file: {outfile2}")

    for k, v in data.items():
        if v['Success']:
            num_success += 1
            length_all_trajs += v['metrics']['traj_length']
            planning_steps_all_trajs += v['metrics']['vlm_steps']

    metrics = {}
    metrics['length_all_trajs'] = length_all_trajs
    metrics['planning_steps_all_trajs'] = float(planning_steps_all_trajs)
    metrics['num_success'] = float(num_success)
    metrics['total_trajs'] = total_trajs
    metrics['avg_traj_length'] = length_all_trajs / total_trajs
    metrics['avg_plan_steps'] = float(planning_steps_all_trajs) / total_trajs

    print(f"Saving file: {outfile}")
    with open(outfile, 'w') as file:
        json.dump(metrics, file, indent=4)
    print(f"Saved file: {outfile}")
