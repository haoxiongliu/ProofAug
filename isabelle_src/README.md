# ProofAug in Isabelle
## Preparations
### PISA Environment Setup
To run ProofAug, please first install Isabelle and run the PISA client. Follow [this repo](https://github.com/haoxiongliu/Portal-to-ISAbelle/tree/dev_lhx) to install Isabelle and compile PISA to make sure the PISA setup is the same with us. 


### Python Environment Setup
Python>=3.11 is needed to run our code. Install the dependencies by
```bash
pip install -r requirements.txt
```

Then, modify all /home/user/... paths in this repository to the corresponding Isabelle paths of yours.


### Prepare an LLM service
You can start a service of deepseek-math-7b-base on localhost:8000 by vllm:
```bash
vllm serve deepseek-ai/deepseek-math-7b-base
```


## Example Script


Set the following environmental values accordingly. For example, 
```bash
base_url="http://localhost:8000/v1"
model="deepseek-ai/deepseek-math-7b-base" 
copy_root=<path_to_save_isa_copies>
pisa_port_base=8100 # [base,base+#checker] should all be available
n_checker=12  # to fewer if your cpu is weaker than ours
```

Then run 

```bash
example="few-shot"   # use null for zero-shot.
template="few-shot"   
python -m main online_chat \
  --dp_processor "proof_aug_1dp" \
  --dst_fp "output/result.jsonl" \
  --src_fp "datasets/minif2f-test.jsonl" \
  --template_fp "templates/$template.json" \
  --example_fp "examples/$example.jsonl" \
  --n_shots 1 \
  --num_try 100 \
  --temperature 0.6 \
  --chat False \
  --comple_sep "concat" \
  --model $model \
  --base_url $base_url \
  --id_key "problem_name" \
  --proof_aug True \
  --use_erp True \
  --port_base $pisa_port_base \
  --copy_root $copy_root \
  --num_checkers $n_checker
```
for the ProofAug+ERP experiment in our paper. Check the result in output/result.jsonl. 

After trying different strategies (modifying the #shots, template and example), you can write a script to get the cumulated proved problems. We provide our cumulated results (66.0% of the problems solved after curation) in cum.tar.gz. Do not distribute this file to prevent it from being included in the training data of LLMs.

