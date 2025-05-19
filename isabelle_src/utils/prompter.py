import random

from utils.misc import jsonl2dps, read_json, group_item_by_key
import random
import re



class MessagesPrompter(object):
    """Need an example_fp and template_fp.
    Template_fp should be a JSON with 'system'(optional), 'assistant' and 'user'.
    Example_fp should be a JSONL contain fields occuring in the template."""

    def __init__(self, template_fp: str, example_fp: str|None = None, 
                 dedup=False, id_key=None, key_map: dict={}):
        self.template = read_json(template_fp)  # type: dict
        self.examples = jsonl2dps(example_fp) if example_fp else []
        self.key_map = key_map  # type: dict
        if key_map:
            # update self.examples with key_map, maintain original key
            for example in self.examples:
                for k, v in key_map.items():
                    if k in example:
                        example[v] = example[k]
        if not id_key:
            id_key_cands = ['xi', 'informal_statement', 'xf', 'formal_statement']
            for key in id_key_cands:
                if key in self.template['user']:
                    id_key = key
                    break
        self.id_key = id_key
        self.dedup = dedup
        self.id2examples = group_item_by_key(self.examples, self.id_key) if self.id_key else None
        self.stop = self.template.get('stop', None)

    def output2dict(self, output: str) -> dict:
        """In fact """
        if m:= re.search(r'\{(.*?)\}', self.template['assistant']):
            output_dict = {m.group(1): output.strip()}
        else:
            raise ValueError(f'Template assistant field is not a single variable.')
        return output_dict

    def get_prompt_msgs(self, data_point: dict, n_shots=1, random_sample=True) -> list[dict]:
        """Return a messages object as final prompt"""
        # 1021 logic update for dedup=True
        messages = []
        if 'system' in self.template:
            messages += [{'role':'system', 'content': self.template['system']}]
        
        if self.dedup and self.id_key:
            cand_examples = []
            exist_ids = set([data_point[self.id_key]])
            for grp_id, grp in self.id2examples.items():
                if grp_id in exist_ids:
                    continue
                exist_ids.add(grp_id)
                selected = random.sample(grp, 1)[0] if random_sample else grp[0]
                cand_examples.append(selected)
        elif self.dedup:
            raise ValueError('Dedup is set but id_key is not provided.')
        else:
            cand_examples = self.examples
        examples = random.sample(cand_examples, n_shots) if random_sample else cand_examples[:n_shots]
        for example in examples:
            messages += [
                {'role': 'user', 'content': self.template['user'].format(**example)},
                {'role': 'assistant', 'content': self.template['assistant'].format(**example)},            
            ]
        messages.append({'role': 'user', 'content': self.template['user'].format(**data_point)})
        return messages

    def get_train_msgs(self, data_point) -> list[dict]:
        messages = []
        if 'system' in self.template:
            messages += [{'role':'system', 'content': self.template['system']}]
        messages += [
            {'role': 'user', 'content': self.template['user'].format(**data_point)},
            {'role': 'assistant', 'content': self.template['assistant'].format(**data_point)},
        ]
        return messages
