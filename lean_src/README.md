# ProofAug in Lean

This repo implements the ProofAug method in Lean.

## Environment Setup

Create a virtual environment and install the dependencies:
```bash
uv venv --python=3.12
uv pip install -r requirements.txt 
```

## Example Script

To evaluate Kimina-Prover-Preview-Distill-1.5B with ProofAug on the benchmark dataset for 1-shot, run:
```bash
dataset="minif2f"
model_root="AI-MO"
model="Kimina-Prover-Preview-Distill-1.5B"
n=1
# baseline
eval_args=(
   -i datasets/$dataset.jsonl
   -m $model_root/$model 
   -o results/$dataset/$model-n$n
   -n $n -g 4 -c 48 -s test
   --template_name kimina
   --use_pty --pty_restart_count 1000
   --memory_limit 10 --timeout 180
)
python eval_pipeline.py "${eval_args[@]}"

# ProofAug
hammer_list=("aesop" "my_hint" "omega")
eval_args=(
   -i datasets/$dataset.jsonl
   -m $model_root/$model 
   -o results/$dataset/$model-n$n-aesop+my_hint+omega
   -n $n -g 4 -c 48 -s test
   --template_name kimina
   --use_pty --pty_restart_count 1000
   --memory_limit 10 --timeout 180
   --proofaug --hammer_list ${hammer_list[@]}
   --use_existing_code results/$dataset/$model-n$n/full_records.jsonl
)
python eval_pipeline.py "${eval_args[@]}"
python main.py get_cumulative_pass "results/minif2f/Kimina-Prover-Preview-Distill-1.5B-n$n*/compilation_summary.csv"
```

There are several cases where due to the bug of repl tactic mode, ProofAug finds an incorrect proof. We have manually checked that suppose we fix the repl bug, ProofAug can find the correct proof in these cases.

## Use ProofAug as an module

ProofAug can be applied to any whole-proof generation model. To use ProofAug as a module, prepare a full_records.jsonl file that contains the "name" field and the "full_code" field, and use "--use_existing_code" in the evaluation script and ignore the model-related arguments.


## Acknowledgement

Part of the architecture of this project is based on [DeepSeek-Prover-V1.5-RL](https://github.com/deepseek-ai/DeepSeek-Prover-V1.5.git) and [Goedel-Prover](https://github.com/Goedel-LM/Goedel-Prover.git). Certain sections of the code have been sourced from DeepSeek-Prover-V1.5-RL, as noted within the code files. The copyright for those sections belongs to [DeepSeek-Prover-V1.5](https://github.com/deepseek-ai/DeepSeek-Prover-V1.5.git) and [Goedel-Prover](https://github.com/Goedel-LM/Goedel-Prover.git).