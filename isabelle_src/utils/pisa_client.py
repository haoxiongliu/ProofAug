from __future__ import print_function

import os
import grpc
import hashlib

from func_timeout import func_set_timeout

import utils.server_pb2 as server_pb2
import utils.server_pb2_grpc as server_pb2_grpc
from utils.misc import UTF2COMMAND
from utils.time_utils import with_timeout
from threading import Lock
import time
import logging
from utils.exceptions import PISAParseError

logger = logging.getLogger(__name__)
logger.addHandler(logging.FileHandler('sv.log'))
logger.setLevel(logging.DEBUG)
logger.handlers[0].setFormatter(logging.Formatter('%(asctime)s:%(levelname)s:%(name)s:%(message)s'))



MAX_MESSAGE_LENGTH = 10485760 * 10
THEOREM_SEPARATOR = "<THM_SEP>"


def create_stub(port=9000):
    channel = grpc.insecure_channel('localhost:{}'.format(port),
                                    options=[('grpc.max_send_message_length', MAX_MESSAGE_LENGTH),
                                             ('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH)])
    return server_pb2_grpc.ServerStub(channel)


def initialise_env(port, isa_path, theory_file_path, working_directory, debug=False):
    return PisaEnv(port=port, isa_path=isa_path, starter_string=theory_file_path, working_directory=working_directory, debug=debug)


class PisaEnv:
    def __init__(self, 
        port=9000, 
        isa_path="/Applications/Isabelle2020.app/Isabelle",
        starter_string="theory Test imports Complex_Main begin",
        working_directory="/Users/qj213/Projects/afp-2021-02-11/thys/Functional-Automata",
        debug=False
    ):
        self.port = port
        self.isa_path = isa_path
        self.starter_string = starter_string
        self.working_directory = working_directory
        self.debug = debug

        self.stub = None
        self.obs_string = None
        self.successful_starting = False
        self.reset()

    def reset(self):
        self.stub = create_stub(port=self.port)
        try:
            print(self.stub.InitialiseIsabelle(server_pb2.IsaPath(path=self.isa_path)).message)
            print(self.stub.IsabelleWorkingDirectory(server_pb2.IsaPath(path=self.working_directory)).message)
            print(self.stub.IsabelleContext(server_pb2.IsaContext(context=self.starter_string)).message)
            self.successful_starting = True
        except Exception as e:
            print("Failure at initialising Isabelle process.\n"
                  "Make sure the path your provide is where the Isabelle executable is.")
            print(e)
        return f"Starting is successful: {self.successful_starting}"

    # seems already legacy
    @func_set_timeout(1800, allowOverride=True)
    def step(self, old_name, step, new_name, delete_old_state=True) -> str:
        '''
        :param old_name: the name of the old state
        :param step: the step to take
        :param new_name: the name of the new state
        :param delete_old_state: if true, delete the old state
        :return: the string of the new state
        I recommend not deleting the default state "default" as it is the starting state.
        '''
        obs_string = "Step error"
        try:
            obs_string = self.stub.IsabelleCommand(
                server_pb2.IsaCommand(command=f"<apply to top level state> {old_name} <apply to top level state> {step} <apply to top level state> {new_name}")).state
            if delete_old_state:
                self.stub.IsabelleCommand(server_pb2.IsaCommand(command=f"<delete> {old_name}"))
                print(f"Deleted old state with name: {old_name}")
        except Exception as e:
            print("***Something went wrong***")
            print(e)
        return obs_string

    def is_finished(self, name_of_tls):
        returned_string = self.post(f"<is finished> {name_of_tls}").strip()
        if returned_string.startswith("t"):
            return True
        else:
            return False

    def reward(self, done):
        if done:
            return 1
        else:
            return 0

    def get_premises(self, name_of_tls, theorem_name, theorem_proof_string):
        message = f"<get dependent theorems>{name_of_tls}<get dependent theorems>{theorem_name}<get dependent theorems>{THEOREM_SEPARATOR}"
        # print(f"Get dependent theroem string: {message}")
        returned_string = self.post(message)
        # print(f"Returned string: {returned_string}")
        premises = returned_string.split(THEOREM_SEPARATOR)
        premises = [premise.strip() for premise in premises]
        # print(premises)
        # print("premises raw", premises)
        # print(f"Returned premises: {'||'.join(premises)}")

        # Function to break down the proof string
        def further_break(chunks, separator=None):
            new_chunks = []
            for chunk in chunks:
                new_chunks.extend(chunk.split(separator))
            return new_chunks

        # Break down the proof string into chunks which might be premises
        possible_premise_chunks = further_break([theorem_proof_string])
        # print("First filter", possible_premise_chunks)
        legit_separators = [",", "(", ")", "[", "]", "{", "}", ":", '"', "<", ">", "\\"]
        for separtor in legit_separators:
            possible_premise_chunks = further_break(possible_premise_chunks, separtor)
        # print("Second filter", possible_premise_chunks)
        possible_premise_chunks = set(chunk.strip() for chunk in possible_premise_chunks)
        # print("Third filter", possible_premise_chunks)
        
        
        # Only include theorems that are in the proof string
        explicit_premises = []
        for premise in premises:
            premise_divisions = premise.split(".")
            for i in range(len(premise_divisions)):
                possible_way_to_refer_to_premise = ".".join(premise_divisions[i:])
                # print("possible_way", possible_way_to_refer_to_premise)
                if possible_way_to_refer_to_premise in possible_premise_chunks:
                    explicit_premises.append(premise)
                    break

        explicit_premises = [premise for premise in explicit_premises if premise.strip()]
        # print(theorem_name, theorem_proof_string, explicit_premises)
        # print("*"*100)
        return explicit_premises

    def get_fact_defintion(self, name_of_tls, fact_name):
        message = f"<get fact definition>{name_of_tls}<get fact definition>{fact_name}"
        # print(f"Get fact definition: {message}")
        returned_string = self.post(message)
        # print(f"Returned definition: {returned_string}")
        return returned_string

    def get_premises_and_their_definitions(self, full_name, only_name, proof_body, debug=False):
        if debug: print("-1")
        self.initialise()
        if debug: print("0")
        # Getting unique name and clone the top level there
        tls_unique_name = str(hashlib.sha256(proof_body.encode("utf-8")).hexdigest())
        tls_unique_name = ''.join(filter(str.isalpha, tls_unique_name))
        # decorated_name = only_name.format(tls_unique_name)
        self.clone_to_new_name(tls_unique_name)
        if debug: print(0.5, "post clone")
        # Substitute proof
        # if not only_name.strip():
        #     sub_proof = f"<allow more time> theorem {proof_body}"
        # else:
        #     sub_proof = f"<allow more time> theorem {full_name}: {proof_body}"
        # if debug: print("1", sub_proof)
        # self.step_to_top_level_state(sub_proof, tls_unique_name, tls_unique_name)
        if debug: print("2, stepping")
        premises = self.get_premises(tls_unique_name, only_name, proof_body)
        if debug: print("3", premises)
        premises_and_their_definitions = [(premise, self.get_fact_defintion(tls_unique_name, premise)) for premise in premises]
        if debug: print("4", premises_and_their_definitions)
        self.post(f"<delete> {tls_unique_name}")
        return premises_and_their_definitions

    # def get_premises_and_their_definitions(self, full_theorem_def, theorem_name, theorem_proof_string):
    #     # print("Get to end: " + self.proceed_until_end_of_theorem_proof(full_theorem_def))
    #     self.initialise()
    #     premises = self.get_premises("default", theorem_name, theorem_proof_string)
    #     # print(premises)
    #     premises_and_their_definitions = [(premise, self.get_fact_defintion("default", premise)) for premise in premises]
    #     return premises_and_their_definitions

    def proceed_until_end_of_theorem_proof(self, theorem_name):
        message = f"<accumulative_step_to_theorem_end> {theorem_name}"
        return self.post(message)

    def accumulative_step_before_theorem_starts(self, theorem_name):
        message = f"<accumulative_step_before_theorem_starts> {theorem_name}"
        return self.post(message)
    
    def accumulative_step_through_a_theorem(self):
        message = f"<accumulative_step_through_a_theorem>"
        return self.post(message)

    @func_set_timeout(1800, allowOverride=True)
    def step_to_top_level_state(self, action, tls_name, new_name):
        # last_obs_string = self.stub.IsabelleCommand(server_pb2.IsaCommand(command=f"<get state> {tls_name}")).state
        obs_string = "Step error"
        try:
            obs_string = self.post(f"<apply to top level state> {tls_name} <apply to top level state> {action} <apply to top level state> {new_name}")
            # print(obs_string)
        except Exception as e:
            print("***Something went wrong***")
            print(e)

        if "error" in obs_string:
            done = False
        else:
            done = self.is_finished(new_name)
        # done = True if ("subgoal" in last_obs_string and "subgoal" not in obs_string) else False
        return obs_string, self.reward(done), done, {}

    def proceed_after(self, line_string):
        return self.post(f"<proceed after> {line_string}") # , forceTimeout=10000

    def initialise(self):
        """Post <initialise> to Isabelle process."""
        return self.post("<initialise>") # , forceTimeout=10

    def clone_to_new_name(self, new_name):
        return self.post(f"<clone> default <clone> {new_name}") # , forceTimeout=10

    # @func_set_timeout(3600, allowOverride=True) # we do not want this timeout, always raise exception
    @with_timeout(300)
    def post(self, action):
        if self.debug: print(action)
        returned = self.stub.IsabelleCommand(server_pb2.IsaCommand(command=action)).state
        if self.debug: print(returned)
        return returned

    def proceed_to_line(self, line_stirng, before_after):
        assert before_after in ["before", "after"]
        try:
            command = f"<proceed {before_after}> {line_stirng}"
            # print(command)
            message = self.stub.IsabelleCommand(server_pb2.IsaCommand(command=command)).state
            # print(message)
        except Exception as e:
            print("Failure to proceed before line")
            print(e)

    def proceed_until_end(self):
        return self.post("<proceed until end>")


class PisaStepEnv(PisaEnv):
    """step env for prove_prm. Should designed for only providing APIs for
    the PisaOneStageServer to interact with Isabelle, rather than implement
    algorithms in its methods."""

    def __init__(self, 
                 port=8000, 
                 isa_path="/home/user/Isabelle2022", 
                 theory_file="/home/user/Isabelle2022/src/HOL/Examples/InterComplex.thy", 
                 working_dir="/home/user/Isabelle2022/src/HOL/Examples/", 
                 debug=False,
                 use_heuristic=True,
                 use_hammer=False,
                 symbol2utf=False):
        super().__init__(port, isa_path, theory_file, working_dir, debug)
        self.initialise()   # initial tls_name is "default"
        self.use_heuristic = use_heuristic
        self.use_hammer = use_hammer
        self.symbol2utf = symbol2utf
        self.lock = Lock()
        self.truncate = False
        # 1105 add mod_simps to heuristics
        self.heuristics = ['by auto', 'by simp', 'by (auto simp: field_simps)', 
                           'by blast', 'by fastforce', 'by (simp add: mod_simps)',
                           'by eval', 'by sos', 'by arith']
        # 0801 'by presburger', 'by arith',  since these two are slow # 0802 linarith to arith
        # 0817 'by force' 'by simp_all' removed
        # head
        extra_libs = ["HOL-Library.Code_Target_Numeral",
                     "HOL-Library.Sum_of_Squares",
                     "HOL-Computational_Algebra.Computational_Algebra",
                     "HOL-Number_Theory.Number_Theory"]
        extra_lib_text = ' '.join([f'"{lib}"' for lib in extra_libs])
        head = f'theory Interactive\nimports HOL.HOL Complex_Main {extra_lib_text}\nbegin\n\n'
        self.root_tls_name = "root"
        self.extra_tlss = set() # dummy, hack for .step for root_tls
        head_res = self.step(head, 'default', self.root_tls_name, assert_success=True)
        self.extra_tlss = set() # in mimic of top_level_state_map in PisaOS
        
    # @func_set_timeout(1800, allowOverride=True)
    # should be legacy. no need to implement algorithm in this class.
    @with_timeout(3600, raise_error=True)  # this timeout return None
    def step(self, step: str, tls_name=None, new_name=None, 
            use_heuristic=None, use_hammer=None, assert_success=False,
            truncate=None, sorry2hammer=True, hammer_retry=1,
            save_fail_tls=False, ret_all_steps=False) -> dict:
        """Compatible with checker.check. 
        Truncate is truncation to the first comment."""
        tls_name = tls_name if tls_name else self.root_tls_name
        new_name = new_name if new_name else f"{tls_name}0s"
        truncate = self.truncate if truncate == None else truncate
        use_heuristic = self.use_heuristic if use_heuristic == None else use_heuristic
        use_hammer = self.use_hammer if use_hammer == None else use_hammer

        mini_steps = self.parse_text(step, sorry2hammer=sorry2hammer)
        tmp_names = [f'{tls_name}__{i}' for i in range(len(mini_steps)-1)]
        tls_names = [tls_name] + tmp_names + [new_name]
        mini_results = []
        first_comment_i = None # need first comment
        for i, mini_step in enumerate(mini_steps):
            if 'normalhammer' in mini_step:
                heuristics = self.heuristics if use_heuristic else ['heuristic_banned']
                for heuristic in heuristics:
                    mini_step_h = mini_step.replace('normalhammer', heuristic)
                    mini_res = self._step(mini_step_h, tls_names[i], tls_names[i+1])
                    if not mini_res['error']:
                        break
                if use_hammer and mini_res['error']:
                    mini_res = self._step(mini_step, tls_names[i], tls_names[i+1], retry=hammer_retry)
                    # considering the corner cases where sorry in the comment, keep use_h on.
                    # raise Exception(f"normalhammer in {mini_step=}, but neither heuristic nor sledgehammer is allowed.")
            else:
                mini_res = self._step(mini_step, tls_names[i], tls_names[i+1])
            mini_results.append(mini_res)
            if mini_res['error']:
                if assert_success: 
                    raise Exception(f"Error in mini_step {i}: {mini_res['error']} in {step=}")
                break
            if truncate and mini_res['done'] and i < len(mini_steps)-1:
                self.clone_to_new_name(tls_names[i+1], new_name) # create tls for new_tls
                break
            if '(*' in mini_step and not first_comment_i:
                first_comment_i = i
        last_mini_res = mini_results[-1]
        if save_fail_tls and last_mini_res['error']:
            self.clone_to_new_name(tls_names[i], new_name)
        if truncate and first_comment_i and first_comment_i > 1 and not last_mini_res['done']:
            mini_results = mini_results[:first_comment_i]
            self.clone_to_new_name(tls_names[first_comment_i], new_name)
        
        if ret_all_steps:
            ret = mini_results
        else:
            ret = mini_results[-1]
            ret['step'] = '\n'.join([s['step'] for s in mini_results])
            for tls_name in tmp_names[:i]:
                self.del_state(tls_name)
        # ret = mini_results if ret_all_steps else ret
        return ret

    @func_set_timeout(1800, allowOverride=False)
    def step_legacy(self, step:str, tls_name='root', new_name='root+default', 
             use_heuristic=None, use_hammer=None, assert_success: bool=False):
        """Return a dict of observation, done, error.
        Note that it the step can be in fact multiple steps.
        TODO: 0729 fix. seems that we must use hammer separately.
        this changes to legacy. the new step will be in fact exec_text.
        We recommand parse_text(text) to steps and then env.step(step).
        For sorry and oops, 
        """
        # last_obs_string = self.stub.IsabelleCommand(server_pb2.IsaCommand(command=f"<get state> {tls_name}")).state
        use_heuristic = self.use_heuristic if use_heuristic == None else use_heuristic
        use_hammer = self.use_hammer if use_hammer == None else use_hammer
        
        if 'normalhammer' in step:
            ret = None
            if use_heuristic:
                for heuristic in self.heuristics:
                    step_h = step.replace('normalhammer', heuristic)
                    ret = self._step(step_h, tls_name, new_name)
                    if not ret['error']:
                        return ret
            if use_hammer:
                ret = self._step(step, tls_name, new_name)
            if not ret:
                # considering the corner cases where soory in the comment, keep use_h on.
                raise Exception(f"normalhammer in {step=}, but neither heuristic and sledgehammer is allowed.")
        else:
            ret = self._step(step, tls_name, new_name)
        
        if assert_success and ret['error']:
            raise ValueError(f'{assert_success=} but encounter error {ret["error"]} at {step=}')

        return ret
    
    def _step(self, step: str, tls_name: str, new_name: str = None, retry=1):
        """Return a dict of step, observation, done, error, is_hammer, tls.
        If you use hammer, make sure the step is the parsed step"""
        if not new_name:
            new_name = f'{tls_name}0s'
        # last_obs_string = self.stub.IsabelleCommand(server_pb2.IsaCommand(command=f"<get state> {tls_name}")).state
        obs = "Step error"
        # context_state = self.get_state(tls_name)
        for i in range(retry):
            try: # if error, new tls_name will not be created.
                # with self.lock:   # different instances will not interfere with each other
                obs = self.post(f"<apply to top level state> {tls_name} <apply to top level state> {step} <apply to top level state> {new_name}")
            except Exception as e:
                print(f"***Something went wrong when apply to top level state {step}***, exception {e}")

            if obs == None:
                # logger.error(f'{step=} returned None observation')
                obs = 'error: probably timeout, but can be other reasons'
            
            # when step is hammer, obs = [hammer strat] <hammer> [ps]
            is_hammer = False
            if '<hammer>' in obs:
                is_hammer = True
                found_step = obs.split('<hammer>')[0].strip()
                obs = obs.split('<hammer>')[1].strip()
            else:
                found_step = step
            error = None
            error_cands = ['error:', "Didn't find top level state", 'Step error', 'Unknown error']
            if any(e in obs for e in error_cands):
                error = obs
                done = False
            else:
                done = self.is_finished(new_name)
            
            # 'success': done, 
            tls = None if error else new_name
            ret = {'step': found_step, 'observation': obs, 'done': done, 
                'error': error, 'is_hammer': is_hammer, 'tls': tls}
            if not error:
                self.extra_tlss.add(new_name)
                if retry > 1:
                    print(f"{found_step=}, retry {i+1}/{retry}")
                break
            else:
                if retry > 1:
                    print(f"Error in step {step}, retry {i+1}/{retry}")
        return ret

    def callATP(self, tls_name: str, use_heuristic=None, use_hammer=None):
        """call ATPs until no subgoal. Return a dict of step, observation, done, error, is_hammer, tls"""
        use_heuristic = self.use_heuristic if use_heuristic == None else use_heuristic
        use_hammer = self.use_hammer if use_hammer == None else use_hammer
        ham_step_lst = []
        ham_tls = tls_name
        while True:
            next_ham_tls = f'{ham_tls}0h'
            ham_res = self.step('sorry', ham_tls, next_ham_tls, 
                                use_heuristic=use_heuristic, 
                                use_hammer=use_hammer)
            if ham_res['error']:
                break
            # ham_step_lst.append(ham_res['step'])
            ham_step_lst.append(ham_res['step'])
            ham_tls = next_ham_tls
            # we dont need to change the tls since it's the same
            if self.tls2type(ham_tls) != 'prove':
            # if 'No subgoals' in self.get_state(ham_tls):
                break                
        ret = ham_res
        ret['step'] = '\n'.join(ham_step_lst)
        return ret

    def parse_text(self, text: str, add_dollar=True, sorry2hammer=True, symbol2utf=None) -> list[str]:
        """parse text into mini_steps. ret_len at least 1."""
        # The hammer and heuristic handling should be out of this env.
        # all hammer calls with sorry, then replace sorry to normalhammer after parsing.
        symbol2utf = self.symbol2utf if symbol2utf == None else symbol2utf
        for (key, value) in UTF2COMMAND.items():
            if symbol2utf:
                text = text.replace(value, key)
            else:
                text = text.replace(key, value)
        text = text.replace('sledgehammer', 'sorry')
        text = text.replace('normalhammer', 'sorry')
        text = text.replace('oops', 'sorry')
        text = text.replace('\\<proof>', 'sorry')

        # try:
        try:
            if add_dollar:  # seems to be a bug in the scala code?
                steps = self.post(f"<parse text> ${text}")
            else:
                steps = self.post(f"<parse text> {text}")
        except Exception as exc: # exception raised by grpc/_channel.py
            err_msg = f'{exc} when parsing {text=}'
            logger.error(err_msg)
            raise PISAParseError(err_msg)
        if steps == None:   # Timeout. Compatibility with the design of with_timeout.
            err_msg = f'Timeout when parsing {text=}. This indicates the PISA server is too busy. This problem should be rechecked.'
            logger.error(err_msg)
            raise PISAParseError(err_msg)
            # logger.error(f'Timeout when parsing {text=}. Returning original text.')
            # steps = [text]  # do not parse when parsing failed.
        steps = steps.split('<SEP>')
        # except Exception as e:
        #     logger.error(f'{traceback.format_exc()}when parsing {text=}')            
        #     steps = [text]
        # steps = [s for s in steps if s.strip() != '']
        # remove weird '$' step and whitespace steps
        # add strip here. why not?
        steps = [s.strip() for s in steps if s != '$' and s.strip() != '']
        if sorry2hammer:
            steps = [s.replace('sorry', 'normalhammer') for s in steps]
        if steps == []:
            steps = [""]
        return steps

    @classmethod
    def obs2type(cls, obs: str) -> str:
        """Return the type of the observation."""
        error_cands = ['error:', "Didn't find top level state", 'Step error', 'Unknown error']
        obs_type = 'unknown'
        if any(e in obs for e in error_cands):
            obs_type = 'error'
        elif 'proof (state)' in obs:
            obs_type = 'state'
        elif 'proof (chain)' in obs:
            obs_type = 'chain'
        elif 'proof (prove)' in obs:
            obs_type = 'prove'
        elif obs == '':
            obs_type = 'empty'
        return obs_type

    def tls2type(self, tls_name: str):
        """Return the type of the top-level state. state, chain, prove, error, empty"""
        obs = self.get_state(tls_name)
        return self.obs2type(obs)

    def clone_to_new_name(self, old_name, new_name):
        res = self.post(f"<clone> {old_name} <clone> {new_name}")
        self.extra_tlss.add(new_name)
        return res # , forceTimeout=10

    def get_state(self, tls_name, assert_exist=True):
        res = self.post(f"<get state> {tls_name}")
        if assert_exist and "Didn't find top level state of given name" in res:
            raise ValueError(f"tls_name {tls_name} not found in isabelle process")
        return res

    def get_level(self, tls_name):
        res = self.post(f"<get_proof_level> {tls_name}")
        return int(res)

    def del_state(self, tls_name: str):
        res = self.post(f"<delete> {tls_name}")
        return res

    def del_states(self, tls_names):
        for tls_name in tls_names:
            self.del_state(tls_name)

    def del_all_extra_tlss(self):
        for tls_name in self.extra_tlss:
            self.del_state(tls_name)
        self.extra_tlss.clear()

    def exit(self):
        try:
            self.post('exit')
        except:
            print("env.post('exit') timed out")
        os.system(f"ps aux | grep {self.isa_path} | awk '{{print $2}}' | xargs kill -9")
        time.sleep(0.1)
