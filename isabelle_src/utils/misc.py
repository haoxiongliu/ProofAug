
import json
import os
import shutil
import psutil
import signal
from collections import defaultdict

UTF2COMMAND = {
    "\u00ac": "\\<not>", "\u00b0": "\\<degree>", "\u00b1": "\\<pm>", 
    "\u00d7": "\\<times>", "\u00f7": "\\<div>",
    "\u03b1": "\\<alpha>", "\u03b2": "\\<beta>", "\u03b3": "\\<gamma>", "\u03b4": "\\<delta>", "\u03b5": "\\<epsilon>", "\u03b7": "\\<eta>", "\u03b8": "\\<theta>",
    "\u03c0": "\\<pi>", "\u03c1": "\\<rho>", "\u03c3": "\\<sigma>", "\u03c4": "\\<tau>", "\u03c6": "\\<phi>",
    "\u2015": "\\<comment>", "\u2022": "\\<bullet>", "\u2039": "\\<open>", "\u203a": "\\<close>",
    "\u2192": "\\<rightarrow>", "\u21d2": "\\<Rightarrow>",
    "\u2200": "\\<forall>", "\u2203": "\\<exists>", "\u2204": "\\<nexists>", "\u2208": "\\<in>", "\u220f": "\\<Prod>",
    "\u2211": "\\<Sum>", "\u221a": "\\<surd>", "\u221e": "\\<infinity>",
    "\u2220": "\\<angle>", "\u2227": "\\<and>", "\u2228": "\\<or>", "\u2229": "\\<inter>", "\u222a": "\\<union>",
    "\u2248": "\\<approx>", "\u2260": "\\<noteq>", "\u2261": "\\<equiv>", "\u2264": "\\<le>", "\u2265": "\\<ge>", "\u22c0": "\\<And>",
    "\u27f9": "\\<Longrightarrow>"
}

def read_from_file(filepath):
    with open(filepath) as f:
        text = f.read()
    return text

def read_json(filepath) -> dict:
    content = read_from_file(filepath)
    return json.loads(content)

def jsonl2dps(filepath, allow_not_exist=True) -> list[dict]:
    if os.path.exists(filepath):
        content = read_from_file(filepath)
        lines = content.strip().split('\n')
        dps = [json.loads(line) for line in lines if line]
    elif allow_not_exist:
        dps = []
    else:
        raise ValueError(f'{filepath} not exists, can not load items from it.')
    return dps

def remove_checked(
    all_dps: list[dict],
    checked_dps: list[dict],
    id_key: str
):
    checked_ids = set([dp[id_key] for dp in checked_dps])
    return [dp for dp in all_dps if dp[id_key] not in checked_ids]

def get_dps_to_run(
    src_fp: str, output_fp: str, 
    start: int = 0, end: int | None = None,
    id_key: str = 'problem',
    rerun: bool = False
) -> list[dict]:
    """remove the items in src_fp that are already in output_fp by id_key"""
    items = jsonl2dps(src_fp, allow_not_exist=False)[start:end]
    if os.path.exists(output_fp):
        if rerun:
            backup_fp = output_fp + '.backup'
            shutil.move(output_fp, backup_fp)
            print(f'{output_fp} already exists and you select rerun. Moving the original data to {backup_fp}')
        else:
            dealt_items = jsonl2dps(output_fp)
            orig_len = len(items)
            items = remove_checked(items, dealt_items, id_key=id_key)
            print(f'{output_fp} already exists!\nOnly generate the remained {len(items)}/{orig_len} items.')

    return items

def write_as_jsonl(items: list[dict], filepath, mode='x', comple_nl=False, verbose=False):
    # in r+ mode, after read, the write will always start at the end 
    if dirpath:=os.path.dirname(filepath):
        os.makedirs(dirpath, exist_ok=True)
    if 'a' in mode and comple_nl:
        with open(filepath, 'r+') as f:
            if f.read()[-1] != '\n':
                f.write('\n')
    with open(filepath, mode) as f:
        for item in items:
            f.write(json.dumps(item)+'\n')
    if verbose:
        print(f'{len(items)} items saved to {filepath}')

def kill_pid_tree(pid):
    try:
        parent = psutil.Process(pid)
        children = parent.children(recursive=True)
        for process in children:
            process.send_signal(signal.SIGTERM)
        parent.send_signal(signal.SIGTERM)
    except psutil.NoSuchProcess as e:
        print(f'process {pid} already killed with exception {e}')

def group_item_by_key(
    items: list[dict],
    key: str,
    return_dict: bool = True
) -> dict[str, list[dict]]:
    """Return a list of items grouped by the key. \n
    set return_dict to True to get a defaultdict
    """
    grouped_items = defaultdict(list)
    for item in items:
        k = item[key]
        grouped_items[k].append(item)
    if return_dict:
        ret = grouped_items
    else:
        # legacy. should avoid as possible
        ret = list(grouped_items.values())
    return ret

def summary_check_output(
    output_fp: str ,
    src_fp: str = 'datasets/xi_tocheck/minif2f-test.jsonl',
    log_fp: str = 'output/summary_check.log',
    id_key: str = 'problem_name',
    pass_k: int | None = None,
    elapsed_time: float | None = None
):
    """Return whether there are multiple dps for each unique i"""
    items = jsonl2dps(output_fp)
    src_items = jsonl2dps(src_fp)
    id_grps = group_item_by_key(items, id_key, return_dict=True)
    # ids = set([item[id_key] for item in items])
    src_ids = set(item[id_key] for item in src_items)
    success_ids = set()
    suc_orig_ids = set()
    suc_ho_ids = set()
    suc_hs_ids = set()
    for id, group in id_grps.items():
        for item in group[:pass_k]:
            if item['success']:
                success_ids.add(id) # for multiple num_samples
            try:
                if item['success_orig']:
                    suc_orig_ids.add(id)
                if item['success_ho']:
                    suc_ho_ids.add(id)
                if item['success_hs']:
                    suc_hs_ids.add(id)
            except:
                pass
    
    logs = []
    logs.append(f'\n{output_fp}')
    logs.append(f'\n{len(items)}/{len(src_items)} data points checked, {len(id_grps)}/{len(src_ids)} distinct, {pass_k=}')
    logs.append(f'\n{len(success_ids)}/{len(id_grps)} succeeds ')
    try:
        logs.append(f'{len(suc_orig_ids)} orig ')
        logs.append(f'{len(suc_ho_ids)} ho ')
        logs.append(f'{len(suc_hs_ids)} hs ')
    except:
        pass
    if elapsed_time: logs.append(f'\n{elapsed_time:.2f} seconds elapsed.')

    log = ''.join(logs)
    print(log)
    with open(log_fp, 'a') as f:
        f.write(log + '\n')

    is_mult = len(items) > len(id_grps) * 1.5
    return is_mult

def sys_prompt_in_user(
    msg: list[dict]
)-> list[dict]:
    """move role:system content to user with \\n\\n"""
    assert msg[0]['role'] == 'system'
    sys_prompt = msg[0]['content']
    msg = msg[1:]
    for item in msg:
        if item['role'] == 'user':
            item['content'] = sys_prompt + '\n\n' + item['content']
    return msg