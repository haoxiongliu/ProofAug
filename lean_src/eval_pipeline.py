import re
import json
import argparse
import asyncio
import os
import pandas as pd
from datetime import datetime
from vllm import LLM, SamplingParams
from prover.lean.verifier import Lean4ServerScheduler
from prover.utils import extract_code, get_semi_proofs, smt_aster, DEFAULT_LEAN_WORKSPACE, LEAN4_DEFAULT_HEADER, DEFAULT_LAKE_PATH, DEFAULT_REPL_PATH
from prover.logger import logger
import random
import torch
from transformers import AutoTokenizer
from os.path import join
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import openai


async def compile_codes(
    codes, cpu, memory_limit, timeout=300, ast=False, tactics=False, use_pty=False, pty_restart_count=3, random_order=False, lean_workspace=DEFAULT_LEAN_WORKSPACE, repl_path=DEFAULT_REPL_PATH, proofaug=False, pa_with_orig=False, args=None
):
    lean4_scheduler = Lean4ServerScheduler(
        max_concurrent_requests=cpu, timeout=timeout, memory_limit=memory_limit, name='verifier', use_pty=use_pty, pty_restart_count=pty_restart_count, lean_workspace=lean_workspace, lake_path=DEFAULT_LAKE_PATH, repl_path=repl_path
    )
    tasks = [{
            "code": code,
            "ast": ast,
            "tactics": tactics,
            "proofaug": proofaug,
            "hammer_type": args.hammer_type,
            "hammer_list": args.hammer_list,
            "pa_with_orig": pa_with_orig
        } for code in codes]
    indexed_tasks = list(enumerate(tasks))
    if random_order:
        random.shuffle(indexed_tasks)
    indices, shuffled_tasks = zip(*indexed_tasks) if indexed_tasks else ([], [])
    try:
        request_id_list = lean4_scheduler.submit_all_request(shuffled_tasks)
        outputs_list = await lean4_scheduler.async_get_all_request_outputs(request_id_list)
        if random_order:
            output_map = {idx: output for idx, output in zip(indices, outputs_list)}
            outputs_list = [output_map[i] for i in range(len(codes))]
        return outputs_list
    except Exception as e:
        logger.error(f"Error compiling codes: {e}")
        raise e
    finally:
        lean4_scheduler.close()

def summarize_results(codes, field):
    df = pd.DataFrame(codes)
    df["correct"] = df.compilation_result.apply(lambda x: x[field])
    df_grp = df.groupby("name")["correct"].sum()
    result = {
        "total": len(df_grp),
        "correct": sum(df_grp > 0),
        "accuracy": f"{sum(df_grp > 0) / len(df_grp) * 100:.2f}",
        "field": field
    }
    return result, df_grp


def main(args):
    # Create output directory first
    # logger.setLevel(logging.DEBUG)
    os.makedirs(args.output_dir, exist_ok=True)
    
    full_records_path = os.path.join(args.output_dir, 'full_records.jsonl')
    
    if args.use_existing_code:
        print(f"Using existing code from {args.use_existing_code}")
        to_inference_codes = []
        with open(args.use_existing_code, 'r') as file:
            for line in file:
                data = json.loads(line)
                name = data["problem_id"] if "problem_id" in data else data["name"]
                to_inference_codes += [{"name": name, "code": code} for code in data["full_code"]]
    elif os.path.exists(full_records_path):
        print(f"Loading existing records from {full_records_path}")
        to_inference_codes = []
        with open(full_records_path, 'r') as file:
            for line in file:
                data = json.loads(line)
                name = data["problem_id"] if "problem_id" in data else data["name"]
                to_inference_codes += [{"name": name, "code": code} for code in data["full_code"]]
    else:
        # Step 1: Inference
        data_list = []
        with open(args.input_path, 'r') as file:
            for line in file:
                data = json.loads(line)
                if args.split is None or (data["split"] == args.split):
                    data_list.append(data)

        model_inputs = []   # for non-remote
        messages_list = []  # for remote
        prefixes = []
        tokenizer_path = args.tokenizer if args.tokenizer else args.model_path
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        if args.template_name:
            with open(join("templates", args.template_name + ".json"), mode='r') as f:
                template = json.loads(f.read())
            template_examples = []
        if args.template_example:
            with open(join("templates", "examples", args.template_example + ".jsonl"), mode='r') as f:
                for line in f:
                    template_examples.append(json.loads(line))
            template_examples = template_examples[:args.n_shot]
        for data in data_list:
            header = data.get('header', LEAN4_DEFAULT_HEADER)
            if args.proofaug and any('smt' in hammer for hammer in args.hammer_list):
                header = "import Smt\nimport Smt.Real\n" + header
            formal_statement = data['formal_statement'] # until := by\n (or no \n, it depends)
            informal_prefix = data.get('informal_prefix', str())
            if informal_prefix:
                if m:= re.match(r"/--(.*?)--/", informal_prefix.strip(), re.DOTALL):
                    problem = m.group(1)
                else:
                    problem = informal_prefix.strip()
            else:
                problem = '[[Informal problem is not available]]'

            # we provide the following fields:
            # problem, informal_prefix, header, formal statement
            # currently do not support few-shot
            if args.template_name:
                messages = []
                if "system" in template and template["system"]:
                    messages.append({"role": "system", "content": template["system"]})
                for example in template_examples:
                    messages.append({"role": "user", "content": template["user"].format(**example)})
                    messages.append({"role": "assistant", "content": template["assistant"].format(**example)})
                
                messages.append({"role": "user", "content": template["user"].format(problem=problem, informal_prefix=informal_prefix, header=header, formal_statement=formal_statement)})            
                messages_list.append(messages)
                prefixes.append(f"{header}{formal_statement}".split(":=")[0])
                # TODO: use model.chat to replace model_inputs
                if args.chat_template_fp:
                    with open(args.chat_template_fp, 'r') as f:
                        chat_template = json.load(f)
                else:
                    chat_template = None

                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    chat_template=chat_template
                )
                model_inputs.append(text)
                
            else:   # TODO: to be legacy by writing a jinja chat_template for this openrlhf template
                model_inputs.append(f"Complete the following Lean 4 code:\n\n```lean4\n{header}{informal_prefix}{formal_statement}")
                prefixes.append(f"{header}{informal_prefix}{formal_statement}".split(":=")[0])

        # find the max length of the model input
        def get_num_tokens(text):
            return len(tokenizer.encode(text))
        max_input_tokens = max([get_num_tokens(input) for input in model_inputs])
        if args.estimate_max_tokens:
            max_model_len = args.max_model_len
            max_tokens = max_model_len - max_input_tokens
            logger.info(f"{max_model_len=}, {max_input_tokens=}, {args.estimate_max_tokens=} so we set {max_tokens=}")
        else:
            max_tokens = args.max_tokens
            max_model_len = max_tokens + max_input_tokens
            logger.info(f"{max_tokens=}, {max_input_tokens=}, {args.estimate_max_tokens=} so we set {max_model_len=}")
        
        # generate the model_outputs
        if args.use_remote_llm:
            client = openai.OpenAI(api_key=args.api_key, base_url=args.base_url)
            e = ThreadPoolExecutor(max_workers=args.max_requests_llm)
            futures = []
            kwargs = {
                "model": args.model_path,
                "max_tokens": max_tokens,
                "temperature": args.temperature,
                "n": args.n,
            }
            if args.top_p > 0:
                kwargs["top_p"] = args.top_p
            
            def post_request(messages):
                try:
                    response = client.chat.completions.create(
                        messages=messages,
                        **kwargs
                    )
                    return [choice.message.content for choice in response.choices]
                    # return response.choices[0].message.content
                except Exception as e:
                    logger.error(f"Error posting request: {e}")
                    return None

            futures = [e.submit(post_request, messages) for messages in messages_list]
            future_to_index = {future: idx for idx, future in enumerate(futures)}
            model_outputs = [None] * len(messages_list)
            for future in tqdm(as_completed(futures), total=len(futures), desc="Generating model outputs"):
                idx = future_to_index[future]
                response_content = future.result()
                if response_content is not None:
                    model_outputs[idx] = response_content
                else:
                    model_outputs[idx] = ["Request failed."] * args.n
        else:
            # TODO: find how can we use the LLM class for chat
            # it seems that model.chat is OK but we need to finish the above
            sampling_params = SamplingParams(temperature=args.temperature, max_tokens=max_tokens, top_p=args.top_p, n=args.n)
            model = LLM(model=args.model_path, seed=1, trust_remote_code=True, swap_space=8, tensor_parallel_size=args.gpu, max_model_len=max_model_len, gpu_memory_utilization=args.gpu_memory_utilization)
            # responses = model.chat(messages_list, sampling_params, use_tqdm=True)
            # model_outputs = [[response.choices[i].message.content for i in range(args.n)] for response in responses]
            vllm_outputs = model.generate(model_inputs, sampling_params, use_tqdm=True)
            model_outputs = [[vllm_output.outputs[i].text for i in range(args.n)] for vllm_output in vllm_outputs]
            del model
            torch.cuda.empty_cache()
        
        to_inference_codes = []
        os.makedirs(args.output_dir, exist_ok=True)
        for i in range(len(data_list)):
            data_list[i]["messages"] = messages_list[i] if args.template_name else model_inputs[i]
            data_list[i]["model_outputs"] = model_outputs[i]
            extract_prefix = str() if args.template_name else model_inputs[i]
            prefix = prefixes[i]
            full_codes = []
            for output in model_outputs[i]:
                model_code = extract_code(extract_prefix + output)
                mc_prefix_end = model_code.find(":=")
                if mc_prefix_end == -1:
                    logger.debug(f"No := found in {extract_prefix + output}")
                    full_code = "[[No := found in the model output]]"
                else:
                    full_code = prefix + model_code[mc_prefix_end:]
                full_codes.append(full_code)
            data_list[i]["full_code"] = full_codes
            name = data_list[i]["problem_id"] if "problem_id" in data_list[i] else data_list[i]["name"]
            to_inference_codes += [{"name": name, "code": code} for code in data_list[i]["full_code"]]
            
            with open(full_records_path, 'a') as f:
                json.dump(data_list[i], f)
                f.write('\n')

    # Step 2: Compile
    codes = [code["code"] for code in to_inference_codes]
    
    print(f"Compiling {len(codes)} codes")
    outputs_list = asyncio.run(compile_codes(
        codes, args.cpu, args.memory_limit, args.timeout, args.ast, args.tactics, 
        args.use_pty, args.pty_restart_count, args.random_order, args.lean_workspace, args.repl_path, args.proofaug, args.pa_with_orig, args))
    for i in range(len(to_inference_codes)):
        to_inference_codes[i]["compilation_result"] = outputs_list[i]

    output_path = f'{args.output_dir}/code_compilation.json'
    os.makedirs(args.output_dir, exist_ok=True)
    with open(output_path, 'w') as json_file:
        json.dump(to_inference_codes, json_file, indent=4)

    # Step 3: Summarize
    result, df_grp = summarize_results(to_inference_codes, args.field)
    summary_path = f'{args.output_dir}/compilation_summary.json'
    hammers = args.hammer_list if args.hammer_list else [args.hammer_type]
    infos = {
        "model": args.model_path,
        "n": args.n,
        "timestamp": datetime.now().strftime("%m%d-%H%M"),
        "hammers": hammers
    }
    result.update(infos)
    with open(args.log_file, "a") as f:
        f.write(f"{result}\n")
    with open(summary_path, "w") as f:
        json.dump(result, f)
    print(f"{summary_path}")
    df_grp.reset_index()[["name", "correct"]].to_csv(summary_path.replace(".json", ".csv"), index=False, header=True, sep='\t', quoting=1, na_rep='Missing')
    print(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_path', type=str, required=True)
    parser.add_argument('-m', '--model_path', type=str, required=True)
    parser.add_argument('-o', '--output_dir', type=str, required=True)
    parser.add_argument('-s', '--split', default=None, type=str)
    parser.add_argument('-n', '--n', default=32, type=int)
    parser.add_argument('-c', '--cpu', default=24, type=int)
    parser.add_argument('-g', '--gpu', default=1, type=int)
    parser.add_argument('-f', '--field', default="complete", choices=["complete", "pass"], type=str)
    parser.add_argument('--tokenizer', type=str, default=None)
    parser.add_argument('--use_remote_llm', action='store_true', default=False)
    parser.add_argument('--max_requests_llm', default=16, type=int)
    parser.add_argument('--template_name', type=str, default=None)
    parser.add_argument('--template_example', type=str, default=None)
    parser.add_argument('--chat_template_fp', type=str, default=None)
    parser.add_argument('--n_shot', type=int, default=1)
    parser.add_argument('--base_url', default=None, type=str)
    parser.add_argument('--api_key', default=None, type=str)
    parser.add_argument('--max_tokens', default=2048, type=int)
    parser.add_argument('--estimate_max_tokens', action='store_true', default=False, help="when set, use max_model_len to deduce max_tokens, otherwise reversely.")
    parser.add_argument('--max_model_len', default=4096, type=int)
    parser.add_argument('--kimina_prompt', action='store_true', default=False)
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--memory_limit', default=10, type=float)
    parser.add_argument('--temperature', default=1.0, type=float)
    parser.add_argument('--top_p', default=0.95, type=float)
    parser.add_argument('--timeout', default=300, type=int)
    parser.add_argument('--gpu_memory_utilization', default=0.9, type=float)
    parser.add_argument('--sync', action='store_true', default=False)
    parser.add_argument('--log_file', default="logs/summary.log", type=str)
    parser.add_argument('--use_existing_code', type=str, default=None)
    parser.add_argument('--ast', action='store_true', default=False)
    parser.add_argument('--tactics', action='store_true', default=False)
    parser.add_argument('--use_pty', action='store_true', default=True)
    parser.add_argument('--nouse_pty', dest='use_pty', action='store_false', default=False)
    parser.add_argument('--hammer_type', type=str, default=None, help="see hint_dict in utils.py for available options")
    parser.add_argument('--hammer_list', nargs='+', default=None)
    parser.add_argument('--proofaug', action='store_true', default=False)
    parser.add_argument('--pa_with_orig', action='store_true', default=False)
    parser.add_argument('--proofaug_legacy', action='store_true', default=False)
    parser.add_argument('--pty_restart_count', default=100, type=int)
    parser.add_argument('--random_order', action='store_true', default=False)
    parser.add_argument('--lean_workspace', type=str, default='mathlib4/')
    parser.add_argument('--repl_path', type=str, default=DEFAULT_REPL_PATH)

    args = parser.parse_args()
    print(args)
    main(args)