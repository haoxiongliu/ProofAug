import time
import os
import shutil
import subprocess
import time
from os.path import join
import fire
import queue
from utils.pisa_client import PisaStepEnv
import logging

logger = logging.getLogger(__name__)
logger.addHandler(logging.FileHandler('main.log'))
logger.setLevel(logging.DEBUG)
logger.handlers[0].setFormatter(logging.Formatter('%(asctime)s:%(levelname)s:%(name)s:%(message)s'))

def init_checker(
    copy_id: int,
    checker_queue: queue.Queue,
    copy_root: str,
    port_base: int,
    use_heuristic: bool = True,
    use_hammer: bool = True,
    wait_server_time: int = 20,
    use_step_env: bool = False,
    isa_path: str = "/home/user/Isabelle2022",
    theory_name: str = "/home/user/InterComplex.thy",  
    jar_path: str = "/home/user/Portal-to-ISAbelle/target/scala-2.13/PISA-assembly-0.1.jar"
):
    port = port_base + copy_id
    os.makedirs(copy_root, exist_ok=True)
    copied_jar_path = join(copy_root, f"pisa_copy{copy_id}.jar")
    if not os.path.exists(copied_jar_path):
        shutil.copy2(jar_path, copied_jar_path)

    copied_isa_root = join(copy_root, f"isabelle_copy_{copy_id}")
    if not os.path.exists(copied_isa_root):
        copy_isa(copied_isa_root, isa_path)
    copied_isa_path = join(copied_isa_root, 'main_isa', os.path.basename(isa_path))
    working_dir = join(copied_isa_path, 'src/HOL/Examples')
    theory_file = join(working_dir, theory_name)
    # since we add try except, kill isa also
    os.system(f"ps aux | grep {copied_isa_path} | grep -v grep | awk '{{print $2}}' | xargs kill -9 > /dev/null 2>&1")
    os.system(f"ps aux | grep {copied_jar_path} | grep -v grep | awk '{{print $2}}' | xargs kill -9 > /dev/null 2>&1")

    jar_pid = subprocess.Popen(["java", "-cp", copied_jar_path, f"pisa.server.PisaOneStageServer{port}"]).pid
    print(f"Server started with pid {jar_pid} at port {port}.")
    time.sleep(wait_server_time)

    if not use_step_env:
        raise ValueError("Legacy Now.")
    else:
        env = PisaStepEnv(port=port, isa_path=copied_isa_path,
                          theory_file=theory_file, working_dir=working_dir,
                          use_hammer=use_hammer, use_heuristic=use_heuristic)
        checker_queue.put(env)
    return jar_pid


def copy_isa(
    copied_isa_root: str,   # like "/data/isa_copy/isabelle_copy_1",
    isa_path: str = "/home/user/Isabelle2022",
    isa_path_user: str = "/home/user/.isabelle"
):
    # when using this function, make sure copied_isa_root does not exist
    os.makedirs(copied_isa_root)

    main_isa_path = os.path.join(copied_isa_root, "main_isa")
    os.makedirs(main_isa_path)
    isa_ver_name = os.path.basename(isa_path)
    shutil.copytree(isa_path, os.path.join(main_isa_path, isa_ver_name), symlinks=True)

    user_isa_path = os.path.join(copied_isa_root, "user_isa")
    shutil.copytree(isa_path_user, user_isa_path, symlinks=True)

    # Edit the settings file such that the user home points to the right directory
    original_isabelle_home_user_string = "$USER_HOME/.isabelle"
    isabelle_home_user_string = str(user_isa_path)
    
    isabelle_settings_path = os.path.join(main_isa_path, isa_ver_name, "etc/settings")
    with open(isabelle_settings_path, "r") as f:
        settings = f.read()
    settings = settings.replace(original_isabelle_home_user_string, isabelle_home_user_string)
    with open(isabelle_settings_path, "w") as f:
        f.write(settings)

if __name__ == '__main__':
    fire.Fire()