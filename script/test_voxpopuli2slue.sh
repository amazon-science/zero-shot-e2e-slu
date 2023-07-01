

project_path="./" # (path to your project)
test_jsonal=${project_path}"processed_data/slue-voxpopuli/slue-voxpopuli_test_blind.csv"
function_path="a path to where put our /eval/evaluate_slue.py"
cd ${function_path}

# For zeroe2eslu
predict_jsonal=${project_path}"results/{seed id in the zeroe2eslu yaml file}/slue-voxpopuli/slue-voxpopuli_slue-voxpopuli-full/gb_inference.jsonl"

# For nlu
#predict_jsonal=${project_path}"results/{seed id in the NLU yaml file}/slue-voxpopuli/slue-voxpopuli_slue-voxpopuli-full/nlu_gb_inference.jsonl"

python evaluate_slue.py -g ${test_jsonal} -p ${predict_jsonal}