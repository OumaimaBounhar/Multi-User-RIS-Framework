import copy
import time
import torch
import multiprocessing as mp

from experiments.store import Store, ExperimentPaths
from experiments.builder import ExperimentBuilder
from experiments.dataset_factory import DatasetMode
from experiments.noise_factory import NoiseMode
from methods.methods import Methods
from methods.test import Test
from reinforcement_learning.deep_q_learning.agent import DeepQLearningAgent


def async_eval_worker(parameters, experiment_root, mode, stop_event):
    params_eval = copy.deepcopy(parameters)

    paths = ExperimentPaths(root=experiment_root)
    store = Store(paths)

    builder = ExperimentBuilder(
        parameters=params_eval,
        store=store
    )

    runner = builder.build(
        dataset_mode=DatasetMode.REUSE,
        noise_mode=NoiseMode.REUSE
    )

    dqn_agent = DeepQLearningAgent(
        parameters=params_eval,
        environment=runner.env,
        paths=store.paths
    )

    # Force async evaluation to CPU
    dqn_agent.device = torch.device(params_eval.async_eval_device)
    dqn_agent.evaluation_q_network.to(dqn_agent.device)
    dqn_agent.target_q_network.to(dqn_agent.device)

    methods = Methods(
        parameters=params_eval,
        channel=runner.channel,
        feedback=runner.feedback,
        probability=runner.probability,
        state=runner.env.state_space,
        Policy_Q=None,
        Policy_network=dqn_agent.evaluation_q_network
    )

    tester = Test(
        parameters=params_eval,
        methods=methods,
        channel=runner.channel,
        probability=runner.probability,
        DQN=dqn_agent
    )

    testing_objects_dict = runner._build_testing_objects_dict(q_policy=None)

    print(f"[ASYNC-EVAL] Worker started for {experiment_root} in mode={mode}")

    while not stop_event.is_set():
        tester.run_model_tests(
            testing_objects_dict=testing_objects_dict,
            checkpoints_dir_ql=store.paths.q_matrices_dir,
            checkpoints_dir_dql=store.paths.dqn_checkpoints_dir,
            mode=mode
        )
        time.sleep(params_eval.async_eval_poll_seconds)

    # Final sweep before exit
    tester.run_model_tests(
        testing_objects_dict=testing_objects_dict,
        checkpoints_dir_ql=store.paths.q_matrices_dir,
        checkpoints_dir_dql=store.paths.dqn_checkpoints_dir,
        mode=mode
    )

    print("[ASYNC-EVAL] Worker stopped")