import copy
import time
import torch
import multiprocessing as mp

from experiments.store import Store, ExperimentPaths
from experiments.builder import ExperimentBuilder
from experiments.runner import Runner
from experiments.dataset_factory import DatasetMode
from experiments.noise_factory import NoiseMode
from methods.methods import Methods
from methods.test import Test
from reinforcement_learning.deep_q_learning.agent import DeepQLearningAgent


def async_eval_worker(
    parameters,
    experiment_root,
    mode,
    stop_event,
    q_policy=None
):
    params_eval = copy.deepcopy(parameters)

    paths = ExperimentPaths(root=experiment_root)
    store = Store(paths)

    builder = ExperimentBuilder(
        parameters=params_eval,
        store=store
    )

    context = builder.build(
        dataset_mode=DatasetMode.REUSE,
        noise_mode=NoiseMode.REUSE
    )

    runner = Runner(context=context)

    dqn_agent = DeepQLearningAgent(
        parameters=params_eval,
        environment=runner.env,
        paths=store.paths
    )

    # Force async evaluation to chosen device (CPU for now)
    dqn_agent.device = torch.device(params_eval.async_eval_device)
    dqn_agent.evaluation_q_network.to(dqn_agent.device)
    dqn_agent.target_q_network.to(dqn_agent.device)

    methods = Methods(
        parameters=params_eval,
        channel=runner.channel,
        feedback=runner.feedback,
        probability=runner.probability,
        state=runner.env.state_space,
        Policy_Q=q_policy,
        Policy_network=dqn_agent.evaluation_q_network
    )

    tester = Test(
        parameters=params_eval,
        methods=methods,
        channel=runner.channel,
        probability=runner.probability,
        DQN=dqn_agent
    )

    testing_objects_dict = runner._build_testing_objects_dict(q_policy=q_policy)

    print(f"[ASYNC-EVAL] Worker started for {experiment_root} in mode={mode}")

    loop_id = 0

    while True:
        if stop_event.is_set():
            print(f"[ASYNC-EVAL] stop_event detected before polling loop {loop_id}")
            break

        print(f"[ASYNC-EVAL] ENTER run_model_tests() | loop={loop_id}")
        tester.run_model_tests(
            testing_objects_dict=testing_objects_dict,
            checkpoints_dir_ql=store.paths.q_matrices_dir,
            checkpoints_dir_dql=store.paths.dqn_checkpoints_dir,
            mode=mode
        )
        print(f"[ASYNC-EVAL] LEFT run_model_tests() | loop={loop_id}")

        if stop_event.is_set():
            print(f"[ASYNC-EVAL] stop_event detected after run_model_tests() | loop={loop_id}")
            break

        print(f"[ASYNC-EVAL] sleeping {params_eval.async_eval_poll_seconds}s before next poll")
        time.sleep(params_eval.async_eval_poll_seconds)
        loop_id += 1

    print("[ASYNC-EVAL] ENTER final sweep before exit")
    tester.run_model_tests(
        testing_objects_dict=testing_objects_dict,
        checkpoints_dir_ql=store.paths.q_matrices_dir,
        checkpoints_dir_dql=store.paths.dqn_checkpoints_dir,
        mode=mode
    )
    print("[ASYNC-EVAL] LEFT final sweep before exit")

    print("[ASYNC-EVAL] Worker exiting")

def start_async_eval_worker(
    parameters,
    experiment_root,
    mode,
    q_policy=None
):
    stop_event = mp.Event()

    process = mp.Process(
        target = async_eval_worker,
        args = (parameters, experiment_root, mode, stop_event, q_policy),
        # daemon=True
        daemon = False
    )
    process.start()

    return stop_event, process