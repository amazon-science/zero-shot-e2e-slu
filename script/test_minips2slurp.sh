project_path="./" # (path to your project)
test_jsonal="a\ path\ to\ download\slurp\ foloder/SLURP/test.jsonl"
function_path="a path to where put our /eval/evaluate.py"
cd ${function_path}

# For zeroe2eslu
predict_jsonal=${project_path}"results/{seed id in the zeroe2eslu yaml file}/slurp/slurp_peoplespeech/gb_inference.jsonl"

# For nlu
#predict_jsonal=${project_path}"results/{seed id in the NLU yaml file}/slurp/slurp_peoplespeech/nlu_gb_inference.jsonl"


python evaluate.py -g ${test_jsonal} -p ${predict_jsonal}