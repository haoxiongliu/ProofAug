from concurrent.futures import as_completed, ThreadPoolExecutor
from dataclasses import dataclass
import queue
import logging
from copy import deepcopy
import random
import time
import traceback
from functools import partial
import re
from os.path import join
from typing import Self


from tqdm import tqdm
import fire

from utils.prompter import MessagesPrompter
from utils.pisa import init_checker
from utils.assistant import post_request
from utils.exceptions import ContextTooLongError, PISAError
from utils.misc import read_json, jsonl2dps, get_dps_to_run, write_as_jsonl, kill_pid_tree, summary_check_output
from utils.pisa_client import PisaStepEnv
from grpc._channel import _InactiveRpcError


logger = logging.getLogger(__name__)
logger.addHandler(logging.FileHandler('sv.log'))
logger.setLevel(logging.DEBUG)
logger.handlers[0].setFormatter(logging.Formatter('%(asctime)s:%(levelname)s:%(name)s:%(message)s'))

@dataclass
class StepNode:
    # the state after yf_p
    yf_p: str = None
    tls: str = None
    i_step: int = None  # finished step index
    ps: str = None # = env.get_state(tls)
    not_skip: int = 0
    step_f: str = None


def proof_aug_1dp(
    data_point: dict,
    idx: int,   # for tls state name
    env_queue: queue.Queue,
    template_fp: str = "templates/comple.json",   # json
    example_fp: str = "examples/dsp-comple.jsonl",    # jsonl
    prompt_dedup: bool = True,
    n_shots: int = 2,
    random_sample: bool = True,
    key_map_name: str = None, # "yf2yf_c",
    num_try: int = 1,
    err_tol: int = 2,
    proof_aug: bool = True,
    use_sorry: bool = True,
    comment_start: bool = False,
    from_offline: bool = False,
    fo_start: int = 0,
    fo_end: int = None,
    use_erp: bool = False,
    sc_add_ps: bool = True,
    **request_kwargs
):
    """required field: yf_p, yf_c. use kay_map for some examples.
    from_offline: use the init_proofs field instead of prompting again.
    """
    key_map = {} if not key_map_name else read_json(f"configs/key_maps/{key_map_name}.json")
    prompter = MessagesPrompter(template_fp, example_fp, dedup=prompt_dedup, key_map=key_map)
    env = None  # env_queue.get()
    try:
        init_proofs = data_point.pop('init_proofs') if from_offline else []
        init_proofs = init_proofs[fo_start:fo_end] if fo_end else init_proofs[fo_start:]
        raw_dp = deepcopy(data_point)
        # needed example fields: xi, yi, xf, yf_c, yf_p, ps
        dsp_kwargs = deepcopy(request_kwargs)
        stop = dsp_kwargs.get('stop', [])
        if not isinstance(stop, list):
            stop = [stop]
        stop = stop + prompter.stop
        dsp_kwargs.update({'stop': stop})

        env = env_queue.get()   # type: PisaStepEnv
        assert type(env) == PisaStepEnv
        root_tls = env.root_tls_name

        # init_proofs = []
        sorry_proofs = []
        xf = raw_dp['xf']
        init_proof = yf = None
        success = False
        success_stage = None # can be init, coarse, fine-grained
        extra_calls = 0
        has_sc = False
        err_count = 0
        
        for i_try in range(num_try):
            has_timeout = False
            try:
                start_tls = f'{idx}p{i_try}t'
                xf_res = env.step(xf, root_tls, start_tls)
                if xf_res['error']:
                    # logger.info(f"{raw_dp} fails to pass compilation")
                    break
                
                # get init_proof
                if not from_offline:
                    dp = deepcopy(raw_dp)
                    ps = env.get_state(start_tls)
                    yf_p = '(* ' if comment_start else ''
                    dp.update({'ps': ps, 'yf_p': yf_p})
                    prompt_msgs = prompter.get_prompt_msgs(dp, n_shots, random_sample)
                    try:
                        content = post_request(prompt_msgs, return_type='content', **dsp_kwargs)
                    except ContextTooLongError as exc:
                        print(f'{exc=}, skipping this')
                        continue
                    init_proof = content # no strip since we use complement
                    init_proofs.append(init_proof)
                else:
                    if i_try >= len(init_proofs):
                        break
                    init_proof = init_proofs[i_try]

                ### get init_sorry
                # infer proof structure simply by proof qed.
                init_steps = env.parse_text(init_proof, sorry2hammer=False)
                if not init_steps[0]:
                    continue
                proof_index = []
                blocks = []
                for i, step in enumerate(init_steps):
                    if step.startswith('proof'):
                        proof_index.append(i)
                    if step == 'qed':
                        try:
                            proof_idx = proof_index.pop()
                        except:
                            break
                        blocks.append((proof_idx, i))

                this_tls = start_tls
                abandon_this = False
                new_steps = deepcopy(init_steps)
                tlss_before = [None]*len(init_steps)  # start tls for each step
                i = 0
                while i < len(new_steps):
                    step = new_steps[i]
                    tlss_before[i] = this_tls
                    if not step:
                        raise ValueError(f"Empty step encountered for {i=}, {new_steps=}, {init_steps=}. This should never happen in current implementation.")
                    next_tls = f'{this_tls}0s'
                    step_res = env._step(step, this_tls, next_tls)
                    if step_res['error']:
                        if not proof_aug:
                            abandon_this = True
                            break
                        if env.tls2type(this_tls) != 'prove': 
                            if 'Illegal application of proof command in' in step_res['error']:            
                                new_steps[i] = ''
                                i += 1
                                continue
                            in_block = False
                            for block in blocks:
                                if block[0] < i and i < block[1]:
                                    for j in range(block[0], block[1]):
                                        new_steps[j] = ''
                                    new_steps[block[1]] = 'sorry'
                                    this_tls = tlss_before[block[0]]
                                    i = block[1]
                                    in_block = True
                                    break
                            if not in_block:
                                abandon_this = True
                                break
                        else:   # in prove mode
                            if step == 'sorry':
                                logger.info(f"Unexpected sorry step error in {i=} of {new_steps}: {step_res['error']}")
                                break    
                            new_steps[i] = 'sorry'
                    else:
                        this_tls = next_tls
                        i += 1
                
                if not use_sorry and 'sorry' in new_steps:
                    abandon_this = True
                if not step_res['done']:
                    abandon_this = True
                if abandon_this:
                    continue

                ### stage 2. determine the sorry sketch proof levels
                parsed_init_proof = '\n'.join(init_steps)
                sorry_proof = '\n'.join([new_step for new_step in new_steps if new_step!= ''])
                init_sorry = deepcopy(sorry_proof)
                while sorry_proof:
                    has_timeout = False
                    if sorry_proof in sorry_proofs:
                        break
                    sorry_proofs.append(sorry_proof)

                    # get the init sorry proof structure
                    has_sc = False
                    steps = env.parse_text(sorry_proof, sorry2hammer=False)
                    tls_now = start_tls
                    tlss_before = []
                    to_break = False
                    for i, step in enumerate(steps):
                        tlss_before.append(tls_now)
                        res = env._step(step, tls_now)
                        if res['error']:
                            if 'Timeout' in str(res['error']):
                                logger.info(f"{res['step']} times out in {sorry_proof=}, {xf=} " + \
                                            "which should have passed compilation in proof_aug process. " + \
                                            "We use sorry to substitute and continue.")
                                res = env._step('sorry', tls_now)
                                has_timeout = True
                            else:
                                logger.info(f"Not timeout Error in {sorry_proof=}, {xf=}, at step {i}: {res['error']}")
                                to_break = True
                                break
                        tls_now = res['tls']
                    if to_break:
                        break
                    tlss_before.append(tls_now)
                    levels_before = [env.get_level(tls) for tls in tlss_before] # len: n+1

                    # call ATP for sorrys
                    for i, step in enumerate(steps):
                        if step != 'sorry': # for less indents
                            continue
                        atp_res = env.callATP(tlss_before[i])
                        level = levels_before[i]
                        if not atp_res['error']:
                            steps[i] = atp_res['step']
                            continue
                        if use_erp:
                            dp = deepcopy(raw_dp)
                            ps = env.get_state(tlss_before[i])
                            if sc_add_ps:
                                yf_p = '\n'.join(steps[:i]) + '\n' + f"(* Proof state:\n{ps} *)\n"
                                hint = ''
                            else:
                                yf_p = '\n'.join(steps[:i]) + '\n'
                                hint = 'proof'
                            send_yf_p = yf_p + hint

                            # do not use the ps template. manually here.
                            dp.update({'yf_p': send_yf_p})
                            prompt_msgs = prompter.get_prompt_msgs(dp, n_shots, random_sample)
                            try:
                                content = post_request(prompt_msgs, return_type='content', **dsp_kwargs)
                                extra_calls += 1
                            except ContextTooLongError as exc:
                                print(f'{exc=}, skipping this')
                                content = None
                            if content:
                                yf_c = hint + str(content) # type: str
                                sc_steps = env.parse_text(yf_c, sorry2hammer=True)
                                expected_state = env.get_state(tlss_before[i+1])
                                this_tls = tlss_before[i]
                                for i_s, sc_step in enumerate(sc_steps):
                                    sc_res = env._step(sc_step, this_tls)
                                    if sc_res['error']:
                                        if env.tls2type(this_tls) != 'prove' or sc_step == 'normalhammer': 
                                            break
                                        else:
                                            sc_res = env.callATP(this_tls)
                                            if sc_res['error']:
                                                break
                                            else:
                                                sc_step_ham = sc_res['step']
                                                sc_steps[i_s] = sc_step_ham
                                    this_tls = sc_res['tls']
                                    this_state = env.get_state(this_tls)
                                    if this_state == expected_state:
                                        break

                                yf_inter = '\n'.join(sc_steps[:i_s+1])
                                if not sc_res['error'] and this_state == expected_state:
                                    has_sc = True
                                    steps[i] = yf_inter
                                    print(f"Successfully proved {yf_inter=} in {dp=}")
                                    continue
                        # we need level-2 to get the last proof (prove) state rather than proof (state)
                        restart_i = max([loc for loc, val in enumerate(levels_before[:i]) if val <= level-2], default=None)
                        end_i = min([loc for loc, val in enumerate(levels_before) if val <= level-2 and loc > i], default=None)
                        if restart_i != None and end_i != None and proof_aug:
                            sorry_proof = '\n'.join(steps[:restart_i] + ['sorry'])
                            if end_i < len(steps):
                                sorry_proof += '\n' + '\n'.join(steps[end_i:])
                            if '{' in steps[restart_i:end_i]:
                                sorry_proof = '' # has {} block in this parent conj.
                        else:
                            sorry_proof = ''
                        # if not continued to next step, break to process next sorry_proof
                        break

                    if 'sorry' not in steps:
                        yf = '\n'.join(steps)
                        if yf == parsed_init_proof:
                            success_stage = 'init_proof'
                        elif sorry_proof == init_sorry:
                            success_stage = 'init_atp'
                        else:
                            success_stage = 'coarse_atp'
                        thm_res = env.step(xf + '\n' + yf)
                        if not thm_res['done']:
                            if 'Timeout' in thm_res['error']:
                                logger.info(f"{thm_res} times out, please check to ensure it's right.")
                                has_timeout = True
                            else:
                                raise ValueError(f"Not-timeout Error in {xf=} {yf=}: {thm_res['error']}")
                        success = True
                        print(f'Successfully proved {xf=}, {yf=} with {init_proof=}')                    
                        break
                
                if success:
                    break
            except Exception as exc:
                err_count += 1
                if err_count > err_tol:
                    raise exc
                error_msg = traceback.format_exc()
                logger.error(error_msg)
                print(error_msg)
                continue            
        env.del_all_extra_tlss()
        env_queue.put(env)
    
    except Exception as exc:
        error_msg = traceback.format_exc()
        logger.error(error_msg)
        print(error_msg)
        env.del_all_extra_tlss()
        if not isinstance(exc, (PISAError, _InactiveRpcError)):
            env_queue.put(env)
        # 1127 add return None to ensure that every data point is correctly handled
        return None

    data_point.update(
        {'init_proof': init_proof, 'sorry_proofs': sorry_proofs, 'yf': yf, 
         'success': success, 'i_try': i_try, 'success_stage': success_stage, 'has_timeout': has_timeout}
    )
    if use_erp:
        data_point.update({'extra_calls': extra_calls, 'has_sc': has_sc})
    return data_point


def online_chat(
    src_fp: str = "datasets/xi_tocheck/minif2f-test.jsonl",
    dst_fp: str = "output/online/debug.jsonl",
    num_workers: int = 8,
    num_checkers: int = 5,
    id_key: str = 'yi', # any value is OK for dst_fp non-exist
    key_map_fp: str = None,
    start: int = 0, end: int | None = None, rerun: bool = False,
    # use_beam_search=True,
    use_hammer: bool = True,
    use_heuristic: bool = True,
    copy_root: str ="/data1/isa_copy/",
    isa_path: str = "/home/user/Isabelle2022",
    jar_path: str = "/home/user/Portal-to-ISAbelle/target/scala-2.13/PISA-assembly-0.1.jar",
    theory_name: str = "InterComplex.thy",
    port_base=8000,
    wait_server_time=20,
    dp_processor: str = 'proof_aug_1dp',
    omit_dp_fp: str = None,
    **dp_kwargs
):
    """handles the preparation works
    10.21: note that default is use_beam_search=True
    """
    # dp_kwargs['use_beam_search'] = use_beam_search
    print(locals())
    dps = get_dps_to_run(src_fp, dst_fp, start, end, id_key, rerun)
    if omit_dp_fp:
        if omit_dp_fp.endswith('.jsonl'):
            omit_dps = jsonl2dps(omit_dp_fp)
            omit_id_keys = [dp[id_key] for dp in omit_dps]
        elif omit_dp_fp.endswith('.json'):
            omit_id_keys = read_json(omit_dp_fp)
        dps = [dp for dp in dps if dp[id_key] not in omit_id_keys]
        print(f'{omit_dp_fp=}, only checking the remained {len(dps)} data points')
    if dps == []:
        print(f"No data points to run")
        return
    if key_map_fp:
        key_map = read_json(key_map_fp)
        for dp in dps:
            for k, v in key_map.items():
                if k in dp:
                    dp[v] = dp.pop(k)
    random.shuffle(dps)

    env_queue = queue.Queue()

    e = ThreadPoolExecutor(num_checkers + 10) # max data points handling, number workers

    futures = []
    dp_proc_func = {'proof_aug_1dp': proof_aug_1dp}[dp_processor]
    # prompters are now determined inside each dp.
    for idx, dp in enumerate(dps):
        futures.append(e.submit(dp_proc_func, dp, idx, env_queue, **dp_kwargs))

    executor_init = ThreadPoolExecutor(20)
    _sub_init = partial(init_checker, checker_queue=env_queue, copy_root=copy_root, port_base=port_base, wait_server_time=wait_server_time, isa_path=isa_path, theory_name=theory_name, jar_path=jar_path, use_hammer=use_hammer, use_heuristic=use_heuristic, use_step_env=True)
    jar_pids_iter_futures = [executor_init.submit(_sub_init, i) for i in range(num_checkers)]

    # future_to_index = {future: idx for idx, future in enumerate(futures)}
    start_time = time.time()
    print(f'{dst_fp=}')
    for future in tqdm(as_completed(futures), total=len(futures)):
        try:
            data_point = future.result()
            print(f'writing to {dst_fp}')
            if data_point:
                write_as_jsonl([data_point], dst_fp, mode='a')
            else:
                print(f"Some response failed for data point ")
        except Exception as exc:
            error_msg = f"{traceback.format_exc()} when handling future"
            logger.error(error_msg)
            print(error_msg)

    e.shutdown(wait=False)
    elapsed_time = time.time() - start_time
    print(f'Results saved to {dst_fp}')
    summary_check_output(dst_fp, src_fp, id_key=id_key, elapsed_time=elapsed_time)

    executor_init.shutdown(wait=False)
    with ThreadPoolExecutor() as executor_exit:
        while not env_queue.empty():
            checker = env_queue.get()
            executor_exit.submit(checker.exit)

    for jar_pid_future in jar_pids_iter_futures:
        kill_pid_tree(jar_pid_future.result())



def recons_yi(yf: str):
    step_str_list = re.findall(r'(?s)(?<=\(\*).*?(?=\*\))', yf)
    yi = ''
    for step_str in step_str_list:
        if '.' not in step_str.strip()[-3:]:
            yi += step_str.strip() + '.\n'
        else:
            yi += step_str.strip() + '\n'
    yi = yi.strip() 
    return yi

# TODO: reimplement this based on that we always stand here and look ahead
def extract_train_dps(
    src_fp: str = 'prompts/isabelle/no_metis_v2.jsonl',
    tgt_dir: str = 'examples/',
    prefix: str = '',
    start: int = 0,
    end: int | None = None,
    assert_success: bool = False,
    check_sorry: bool = False,
    use_hammer: bool = False,
    use_heuristic: bool = False,
    isa_path: str = "/home/user/Isabelle2022",
    theory_name: str = "InterComplex.thy",
    port=8199,
    mode='x',
    reconstruct_yi: bool = False,
    step_tol: int | None = None,
    from_prompt: bool = True,
):
    """For simplicity, only use one ISA server
    Note that the extract logic is different from inference time.
    TODO: the prompt files switch to examples that satisfy MessagesPrompter.
    mini_step level: xi, yi, xf, yf_p, yf_woi_p, ps, mini_step
    proof_state level: xi, yi, xf, yf_p, yf_woi_p, ps, ps_step (I prefer this, but need sv_wi)
    step level: xi, yi, xf, yf_p, yf_woi_p, ps, step_i, step_f
    """

    items = jsonl2dps(src_fp)
    items = items[start: end]

    filter_items = []
    for item in items:
        if 'success' in item.keys():
            if not item['success']: # can be optimized using .get()
                continue
        if 'contradict' in item.keys():
            if item['contradict']:
                continue
        if step_tol:
            num_step = len(extract_steps_yi(item['steps_yi']))
            if num_step - item['not_skip'] > step_tol:
                continue
        filter_items.append(item)
    items = filter_items
    print(f'Now processing {len(items)} data points of proofs')
    if not items:
        raise ValueError(f"No valid data points in {src_fp}")

    working_dir = join(isa_path, 'src/HOL/Examples')
    theory_file = join(working_dir, theory_name)
    env = PisaStepEnv(port, isa_path, theory_file, working_dir, use_hammer=use_hammer, use_heuristic=use_heuristic)

    # target data
    # step means one comment to next comment
    # yf: steps with comment
    # yf_woi: steps without comment

    whole_examples = [] # xiyixf -> yf 
    # baldur = [] # xf -> yf_woi
    mini_examples = [] # xfyf_woi[:i+1]ps[i] -> yf_woi[i]
    ps_examples = []    # original ps
    ps_wi_examples = []
    step_examples = [] # xixfyf[:i+1]ps[i] -> yf[i]
    step_trunc_examples = []
    comple_woi_exas = []
    comple_exas = []
    
    # sv_wi = [] # xiyixfyf[:i+1]ps[i] -> yf[i]
    
    for i_prob, item in tqdm(enumerate(items), total=len(items)):
        xi_key = 'xi' if 'xi' in item.keys() else 'informal_statement'
        xf_key = 'xf' if 'xf' in item.keys() else 'formal_statement'
        yi_key = 'yi' if 'yi' in item.keys() else 'informal_proof'
        yf_key = 'yf' if 'yf' in item.keys() else 'formal_proof'
        xi, xf = item[xi_key], item[xf_key]
        yi, yf = item[yi_key], item[yf_key]
        if reconstruct_yi:
            yi = recons_yi(yf)
        xfyf = xf + '\n' + yf

        root_tls = env.root_tls_name
        start_tls = f'{i_prob}s'
        xf_res = env.step(xf, root_tls, start_tls, assert_success=assert_success)
        if xf_res['error']:
            # logger.error(f"Failed to complile {xf}")
            continue
        # first get steps_i and steps_f
        mini_steps = env.parse_text(yf, sorry2hammer=False) + ['(* Q.E.D. *)']  # dummy step
        if mini_steps[0] == 'proof -': # and is_comment(mini_steps[1])
            this_tls = start_tls + '0'
            res = env.step(mini_steps[0], start_tls, this_tls, assert_success=assert_success)
            if res['error']:
                logger.error(f"Failed to complie {mini_steps[0]}")
                continue
            current_step = StepNodeTrain(this_tls, None, None)
            yf_woi_p = yf_p = mini_steps[0] + '\n'
            # yf_p += mini_steps[1] + '\n'
            mini_steps = mini_steps[1:]
        else:
            current_step = StepNodeTrain(start_tls, None, None)    # only for sv
            this_tls = start_tls
            yf_woi_p = yf_p = ''
        ps = env.get_state(this_tls)
        step_ps = ps_ps = ps_wi_ps = mini_ps = ps
        step_yf_p = ps_wi_yf_p = yf_p
        ps_yf_woi_p = mini_yf_woi_p = yf_woi_p
        step_nodes = [] #type: list[StepNodeTrain]
        ps_step_minis = []
        ps_wi_step_minis = []
        
        # TODO: step_yf_p and ps_yf_p merge ps state newlines? not valuable.
        for i_step, mini_step in enumerate(mini_steps):
            mini_step = mini_step.strip()
            is_comm = is_comment(mini_step)
            # start new step if is_comm
            if is_comm:
                step_i = current_step.step_i    # last step step_i
                step_f = current_step.construct_step_f()
                # the assistant only needs to answer step_f. by EOS.
                if step_i:
                    step_examples.append(dict(xi=xi, xf=xf, yi=yi, yf_p=step_yf_p, 
                                            step_i=step_i, ps=step_ps, step=step_f))

                    step_nodes.append(current_step)
                current_step = StepNodeTrain(this_tls, current_step, step_i = mini_step)
                step_ps, step_yf_p = ps, yf_p   # still last step yf_p

                # trunc step. need yf_c
                if i_step != len(mini_steps)-1: # exclude dummy step
                    # from i_step+1 to i_step now.
                    step_c = '\n'.join(mini_steps[i_step+1:-1])
                    step_trunc_examples.append(dict(xi=xi, xf=xf, yi=yi, yf_p=step_yf_p, 
                                                step_i=mini_step, ps=step_ps, step=step_c))

            # update mini_step state
            next_tls = this_tls + '0'   # for branches 0123...
            step_res = env.step(mini_step, this_tls, next_tls, 
                                sorry2hammer=check_sorry)
            if step_res['error']:
                error_info = f"Failed to extract proof with {step_res['error']} at {mini_step} in {xfyf=}"
                logger.error(error_info)    # in principle should all be correct
                if assert_success: raise ValueError(error_info)

            # state and yf_p is updated here!!!
            last_ps = ps
            ps = step_res['observation']
            last_yf_p = deepcopy(yf_p)
            yf_p += mini_step + '\n'
            if not is_comm:
                yf_woi_p += mini_step + '\n'

            # mini_step & current_step step_f
            if not is_comm:
                current_step.mini_states.append(mini_ps)
                current_step.mini_steps.append(mini_step)   # construct step_f
                # mini_step (default woi)
                mini_examples.append(dict(xf=xf, yf_p=mini_yf_woi_p, ps=mini_ps, step=mini_step))
                mini_ps, mini_yf_woi_p = ps, yf_woi_p
                
                yf_woi_c = '\n'.join([_m_step for _m_step in mini_steps[i_step:-1] if not is_comment(_m_step)])
                comple_woi_exas.append(dict(xi=xi, xf=xf, yi=yi, yf_p=mini_yf_woi_p, 
                                            ps=last_ps, yf_c=yf_woi_c))
            
            yf_c = '\n'.join(mini_steps[i_step:-1])
            comple_exas.append(dict(xi=xi, xf=xf, yi=yi, yf_p=last_yf_p, 
                                        ps=last_ps, yf_c=yf_c))

            # ps_step 
            if not is_comm:
                ps_step_minis.append(mini_step)
                if 'proof (state)' in step_res['observation']:
                    ps_step = ' '.join(ps_step_minis)
                    ps_examples.append(dict(xf=xf, yf_p=ps_yf_woi_p, 
                                            ps=ps_ps, step=ps_step))
                    ps_step_minis = []
                    ps_ps, ps_yf_woi_p = ps, yf_woi_p
            
            # ps_wi_step
            ps_wi_step_minis.append(mini_step)
            if 'proof (state)' in step_res['observation']:    
                ps_wi_step = ' '.join(ps_wi_step_minis)
                ps_wi_examples.append(dict(xi=xi, xf=xf, yi=yi, yf_p=ps_wi_yf_p, 
                                           ps=ps_wi_ps, step=ps_wi_step))
                ps_wi_step_minis = []
                ps_wi_ps, ps_wi_yf_p = ps, yf_p

            this_tls = next_tls
        # no need to add one more step. we have added 'Q.E.D.' dummy step           
        # step_nodes: do not contain ps, only context_tls, step_i, step_f
        # construct: xi, xf, yi, yf_p=step_yf_p, step_i=step_i, ps=step_ps, step=step_f
        # dtv & baldur, yf_p="" to be compatible with proof_aug_1dp
        whole_examples.append(dict(xi=xi, xf=xf, yi=yi, yf=yf, yf_woi=yf2yf_woi(yf), yf_p=""))
        # baldur.append((template['baldur'].format(xf=xf), yf2yf_woi(yf)))

    write_as_jsonl(comple_exas, join(tgt_dir, f'{prefix}comple.jsonl'), mode)
    write_as_jsonl(comple_woi_exas, join(tgt_dir, f'{prefix}comple_woi.jsonl'), mode)
    write_as_jsonl(step_trunc_examples, join(tgt_dir, f'{prefix}step_trunc.jsonl'), mode)
    write_as_jsonl(whole_examples, join(tgt_dir, f'{prefix}whole.jsonl'), mode)
    write_as_jsonl(step_examples, join(tgt_dir,f'{prefix}step.jsonl'), mode)
    write_as_jsonl(ps_examples, join(tgt_dir,f'{prefix}ps.jsonl'), mode)
    write_as_jsonl(ps_wi_examples, join(tgt_dir,f'{prefix}ps_wi.jsonl'), mode)
    write_as_jsonl(mini_examples, join(tgt_dir, f'{prefix}mini.jsonl'), mode) # gptf
    print('Finished extracting examples!')


def extracted2messages(
    src_fp: str = 'output/examples/numina_v1-0/step.jsonl',
    tgt_fp: str = 'datasets/numina_v1-0-step-sv/all.jsonl',
    # example_fp: str | None = 'examples/df-step.jsonl',
    template_fp: str = 'templates/sv.json',
    mode: str = 'x',
):
    templater = MessagesPrompter(template_fp)
    dps = jsonl2dps(src_fp)
    ds_dps = [{'messages': templater.get_train_msgs(dp)} for dp in dps]
    write_as_jsonl(ds_dps, tgt_fp, mode)


class StepNodeTrain:

    def __init__(self, context_tls: str, parent: Self | None, step_i: str | None) -> None:
        self.parent = parent
        self.context_tls = context_tls
        self.ps = None
        self.mini_steps = []
        self.mini_states = []   # states before ministep
        self.ps_steps = []
        self.ps_states = []
        # self.step_f = None      # do not contain comment
        self.step_i = step_i    # contain (* . a mini step.

    @property
    def step_f(self) -> str:
        return self.construct_step_f()

    def construct_step_f(self) -> str:
        """construct step_f from mini_steps and mini_states"""
        step_f = '\n'.join(self.mini_steps) +'\n'
        # self.step_f = step_f
        return step_f

def is_comment(mini_step: str) -> bool:
    return '(*' in mini_step and '*)' in mini_step

def yf2yf_woi(yf: str) -> str:
    """remove the comment from yf"""
    return re.sub(r'(?s)\(\*.*?\*\)\n*', '', yf)

def comment2step(step_i:str) -> list[str]:
    """remove the (* and *)"""
    return step_i.split('(*')[-1].split('*)')[0].strip()


def extract_steps_yi(steps_yi: str) -> list[str]:
    """steps_yi in the format **Step 1:**..."""
    step_i_lst = re.split(r'\*\*Step \d+:\*\*', steps_yi)
    step_i_lst = [f'(* {step.strip()} *)' for step in step_i_lst if step.strip()]
    return step_i_lst

def extract_steps(yi: str, qw_hack: bool = False, add_comment: bool = False, spm_type: str = 'rule') -> list[str]:
    """split yi"""
    # first version has |.\$\n as the separator
    # (?!\\end)
    if spm_type == 'rule':
        steps = re.split(r'(?<=\S\S\.\s|\.\$\$\s|.\.\$\s|\.\\\]\s)', yi)
    elif spm_type == 'fg':  # finegrained version
        pattern = r'(?<=\S\S\.\s|\.\$\$\s|.\.\$\s|\.\\\]\s)|\n\n'
        steps = re.split(pattern, yi)
    else:
        raise ValueError(f"Invalid spm_type: {spm_type}")
    step_lst = []
    for i, step in enumerate(steps):
        # Hack for numina
        omit_kw = ['copyright', 'original thread'] #  'above solution',
        for kw in omit_kw:
            if kw in step.lower():
                step = ''
                break
        # numina authors
        if re.search(r'~[a-zA-Z]', step):
            continue
        # latex end
        if m:= re.match(r'(?s)(\\end\{\S+?\})(.*)$', step):
            m: re.Match
            last, step = m.group(1), m.group(2)
            if i > 0:
                step_lst[-1] = step_lst[-1] + last
        if not step.strip():
            continue
        step_lst.append(step.strip())

    # hack for Qwen answers
    if qw_hack:
        step0 = step_lst[0] # type: str
        if step0.startswith('To'):
            step_lst = step_lst[1:]
            step_lst[0] = step0 + '\n' + step_lst[0]

    if add_comment:
        step_lst = [f'(* {comment2step(step)} *)' for step in step_lst]

    return step_lst


if __name__ == '__main__':
    fire.Fire()