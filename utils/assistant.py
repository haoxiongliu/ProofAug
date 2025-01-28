from concurrent.futures import as_completed, ThreadPoolExecutor
from dataclasses import dataclass, asdict, replace, field
import json
import re
import random
import warnings
from tqdm import tqdm
from time import sleep
from copy import deepcopy
import fire
from jinja2 import Template

from copy import deepcopy
from openai import OpenAI
from openai.types.chat import ChatCompletion
from openai.types.completion import Completion
from utils.misc import write_as_jsonl, read_json, read_from_file, sys_prompt_in_user, get_dps_to_run
from utils.prompter import MessagesPrompter
import logging
from utils.time_utils import time_logger
from utils.exceptions import ContextTooLongError


logger = logging.getLogger(__name__)
logger.addHandler(logging.FileHandler('assistant.log'))
logger.setLevel(logging.DEBUG)
logger.handlers[0].setFormatter(logging.Formatter('%(asctime)s:%(levelname)s:%(name)s:%(message)s'))


# can use asdict() function to transform dataclass instance into a dict
@dataclass
class RequestParameters:
    """Set the default params for openai API"""
    model: str | None = None
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    temperature: float = 0.0
    max_tokens: int = 2048
    top_p: float = 0.95
    stream: bool = False
    logprobs: int | None = None
    timeout: int = 600
    n: int = 1
    stop: list[str] = field(default_factory=lambda: [])
    extra_body: dict = field(default_factory=lambda: dict(skip_special_tokens=False))


class Assistant():
    
    def __init__(self, api_key: str = 'foo', base_url: str = "http://localhost:8000/v1", 
                 system_prompt: str | list[dict] = [], chat: bool = True, comple_sep: str = 'newline', 
                 **request_kwargs) -> None:
        """Use chat=False for Completions."""
        self.parameters = RequestParameters() # we do not pass args here
        self.api_key = api_key
        self.timeout_length = 10
        self.last_err_response = None
        self.log_fp = None
        self.base_url = base_url
        self.comple_sep = comple_sep # in newline, xml
        self.chat = chat
        if not chat:
            default_stop = sep2stop(comple_sep)
            if 'stop' not in request_kwargs:
                request_kwargs['stop'] = default_stop
        self.set_request_parameters(**request_kwargs)

        if isinstance(system_prompt, str):
            self.messages = [{"role": "system", "content": system_prompt}]
        elif isinstance(system_prompt, list):
            self.messages = deepcopy(system_prompt)
        else:
            raise ValueError("system_prompt for assistant should be str or list")
        

    def substitute_msg(self, messages: list, copy=True):
        self.messages = deepcopy(messages) if copy else messages

    def switch_to_in_user(self):
        """move role:system content to user with \n\n"""
        self.messages = sys_prompt_in_user(self.messages)

    def set_request_parameters(self, **kwargs):
        extra_group = {'skip_special_tokens', 'use_beam_search', 'include_stop_str_in_output'}
        extra_body = {k: v for k, v in kwargs.items() if k in extra_group}
        for k in extra_group:
            kwargs.pop(k, None)
        self.parameters = replace(self.parameters, **kwargs)
        self.parameters.extra_body.update(extra_body)

    def msg2prompt(self, messages=None, comple_sep=None) -> str:
        messages = self.messages if not messages else messages
        comple_sep = self.comple_sep if not comple_sep else comple_sep
        prompt = msg2prompt(messages, comple_sep)   # careful with the name conflict
        return prompt


    # @retry(wait=wait_random_exponential(min=1, max=10), stop=stop_after_attempt(30))
    def send_request(self, user_prompt=None, add_to_msg: bool = True, wait_time=30, max_retry=3, timeout=600, return_response=False, pure_add: bool = True) -> str:
        if user_prompt:
            self.receive_user_prompt(user_prompt)
        request = asdict(self.parameters)
        request['timeout'] = timeout
        if request['model'] == None:
            raise ValueError('Please specify the model name')
        client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        for i in range(max_retry):
            try:
                if self.chat:
                    request["messages"] = self.messages
                    response = client.chat.completions.create(**request)
                    response_message = response.choices[0].message.model_dump()
                else:
                    request["prompt"] = self.msg2prompt()
                    response = client.completions.create(**request)
                    response_message = {'role': 'assistant', 'content': response.choices[0].text}
                break
            except Exception as e:
                print(f"{i}th try request, exception: {e}")
                if i == max_retry - 1:
                    logger.error(f"Retry times exceed {max_retry} times with exception {e}")
                    raise IOError(f"Request failed after {max_retry} times")
                if m:= re.search(r'maximum context length is (\d+) tokens. However, you requested \d+ tokens \((\d+) in the messages', str(e)):
                    max_context_len = int(m.group(1))
                    query_len = int(m.group(2))
                    if query_len < max_context_len:
                        max_new_len = max_context_len - query_len
                        print(f'Adjusting max_tokens from {max_context_len} to {max_new_len} and retrying.')
                        request['max_tokens'] = max_new_len
                        continue
                    else:
                        raise ContextTooLongError(f"{query_len=} is greater than {max_context_len=}. Please check the input messages.")
                        # return None
                sleep(wait_time)
                continue
        
        try:
            if add_to_msg:
                if pure_add:
                    response_message = {key: response_message[key] for key in ['role', 'content']}
                self.messages.append(response_message)
            content = response_message['content']
        except Exception as e:
            warnings.warn(f"Exception {e} encountered when extract content from the response\n{response_message}")
            content = None
        return response if return_response else content
    
    def receive_user_prompt(self, user_prompt: str):
        message = {"role": "user", "content": user_prompt}
        self.messages.append(message)
        return

    def set_log_fp(self, log_fp):
        self.log_fp = log_fp
        return

    def save_log(self, overwrite=True):
        mode = 'w' if overwrite else 'a'
        with open(self.log_fp, mode) as f:
            f.write(json.dumps(self.messages)+'\n')
    
    def reset_dialogue(self, remain_system_prompt=True):
        if self.messages and remain_system_prompt and self.messages[0]["role"] == 'system':
            self.messages = [self.messages[0]]
        else:
            self.messages = []


def msg2prompt(messages, comple_sep='newline') -> str:
    """comple_sep can be newline, user, xml.
    TODO: add jinja template
    use three newlines before role: user for newline"""
    if comple_sep == 'newline':
        prompt = ''
        for msg in messages:
            role, content = msg['role'], msg['content']
            if role == 'user' and prompt != '':
                prompt += '\n\n\n'
            prompt += content
    elif comple_sep == 'xml':
        prompt = ''
        for msg in messages:
            role, content = msg['role'], msg['content']
            prompt += f'<{role}>\n{content}\n</{role}>\n\n'
        prompt += '<assistant>\n'
    elif comple_sep == 'concat':
        prompt = ''.join([message['content'] for message in messages])
    elif comple_sep == 'chatml':
        # tokenizer = AutoTokenizer.from_pretrained('deepseek-ai/deepseek-math-7b-base')
        template = Template(read_from_file('templates/chatml.jinja'))
        prompt = template.render(messages=messages, add_generation_prompt=True)
        # prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:   # for jinja templates
        template = Template(read_from_file(comple_sep))
        prompt = template.render(messages=messages, add_generation_prompt=True)
        # prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    return prompt


def sep2stop(comple_sep) -> list[str]:
    """create a sep2stop_dict for different comple_sep"""
    sep2stop_dict = {
        'newline': ["\n\n\n"],
        'xml': ["</assistant>"],    # \n removed
        'concat': [],
        'chatml': ["<|im_end|>"]
    }
    # if comple_sep == 'concat':
    #     print(f'You have set {comple_sep=}, please make sure to set the stop')
    #     # print(f'{comple_sep=} not valid, now only support [newline, xml, concat]')
    stop = sep2stop_dict[comple_sep]
    return stop


def post_request(
    messages: list[dict], 
    base_url: str = 'http://localhost:8000/v1', api_key: str = 'foo', 
    wait_time=30, max_retry=3, chat=True, comple_sep='newline',
    prompt_in_user: bool = False, return_type: str = 'response',
    **request_params
) -> ChatCompletion | Completion | str:
    """Return server response, can be ChatCompletion or Completion.
    transform use_beam_search to extra_body.
    Use choice.message.content and choice.text respectively.
    Consult vllm api documentation for request_params.
    Important params: model, n"""
    # if 'use_beam_search' in request_params:
    #     request_params['extra_body'] = {'use_beam_search': request_params['use_beam_search']}
    #     del request_params['use_beam_search']
    assistant = Assistant(api_key=api_key, base_url=base_url, chat=chat, comple_sep=comple_sep, **request_params)
    assistant.substitute_msg(messages, copy=True)
    if prompt_in_user:
        assistant.switch_to_in_user()
    response = assistant.send_request(wait_time=wait_time, max_retry=max_retry, add_to_msg=False, return_response=True)

    if return_type == 'response':
        ret = response
    elif return_type == 'content':
        choice = response.choices[0]
        ret = choice.message.content if chat else choice.text
        
    return ret


# almost gen_sol, but support various templates and examples
def offline_chat(
    src_fp: str,
    dst_fp: str,
    template_fp: str,   # json
    example_fp: str,    # jsonl
    num_workers: int = 256,
    n_shots: int = 2,
    id_key: str = 'xi', # any value is OK for dst_fp non-exist
    save_msgs: bool = True,
    grp_key: str | None = None, # for single input field, such as xi
    dedup: bool = False,
    start: int = 0, end: int | None = None, rerun: bool = False,
    n_prompts: int = 1,
    key_map_fp: str | None = None, # 'configs/key_maps/xxx.json'
    add_yf_p: bool = False, # ad-hoc ablation experiment for proofaug
    **request_kwargs
):
    """offline chat with the model. Each data point updates the user and messages field"""
    print(locals())
    import time
    start_time = time.time()
    dps = get_dps_to_run(src_fp, dst_fp, start, end, id_key, rerun)
    if add_yf_p:
        for dp in dps:
            dp['yf_p'] = ""
    # dps = jsonl_2_items(src_fp)
    random.shuffle(dps)
    if key_map_fp:
        key_map = read_json(key_map_fp)
        for dp in dps:
            for k, v in key_map.items():
                if k in dp:
                    dp[v] = dp.pop(k)

    chat = request_kwargs['chat']
    e = ThreadPoolExecutor(num_workers)
    
    prompter = MessagesPrompter(template_fp, example_fp, dedup)
    stop = request_kwargs.get('stop', [])
    if not isinstance(stop, list):
        stop = [stop]
    stop = stop + prompter.stop
    req_kw_actual = deepcopy(request_kwargs)
    req_kw_actual.update({'stop': stop})

    futures = []
    future_to_index = dict()
    for i, dp in enumerate(dps):
        for _ in range(n_prompts):
            messages = prompter.get_prompt_msgs(dp, n_shots)
            future = e.submit(post_request, messages=messages, **req_kw_actual)
            future_to_index[future] = i
            futures.append(future)

    print(f'{dst_fp=}')
    for future in tqdm(as_completed(futures), total=len(futures)):
        try:
            # in fact only support chat=True?
            response = future.result()  # type: ChatCompletion | Completion
            idx = future_to_index[future]
            dp = dps[idx]   # type: dict
            output_dps = []
            for i, choice in enumerate(response.choices):
                output = choice.message.content if chat else choice.text
                output_dict = prompter.output2dict(output)
                output_dp = deepcopy(dp)
                output_dp.update(output_dict) # e.g. {'mini_step': 'by auto'}
                if save_msgs:
                    output_dp.pop('messages', None)
                    output_dp['messages'] = prompter.get_train_msgs(output_dp)
                output_dps.append(output_dp)
            write_as_jsonl(output_dps, dst_fp, mode='a')

        except Exception as exception:
            logger.error(f'Exception {exception} occurred in offline chat.')
            print(f'Exception {exception} occurred.')
    e.shutdown(wait=True)
    print(f'Results saved to {dst_fp}')
    elapsed_time = time.time() - start_time
    time_logger.info(f'offline chat of {dst_fp=} finished using {elapsed_time:.2f}s.')



if __name__ == '__main__':
    fire.Fire()