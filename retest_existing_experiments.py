import argparse
import ast
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from config.parameters import Parameters
from systemModel.codebooks import CodebookSpec

from experiments.store import Store, ExperimentPaths
from experiments.builder import ExperimentBuilder
from experiments.dataset_factory import DatasetMode
from experiments.noise_factory import NoiseMode
from experiments.runner import Runner

from reinforcement_learning.deep_q_learning.agent import DeepQLearningAgent
from reinforcement_learning.q_learning.utils import load_Policy
from reinforcement_learning.deep_q_learning.components.seed import set_seed


SECTION_COMMON = "=== COMMON PARAMETERS ==="
SECTION_DQN = "=== DQN PARAMETERS ==="
SECTION_QL = "=== Q-LEARNING PARAMETERS ==="


def _parse_scalar(value: str) -> Any:
    value = value.strip()
    if value == "None":
        return None
    if value == "True":
        return True
    if value == "False":
        return False

    try:
        return ast.literal_eval(value)
    except Exception:
        return value


def _parse_codebook_specs(raw: Any) -> Optional[List[CodebookSpec]]:
    if raw is None:
        return None

    if isinstance(raw, list):
        if all(isinstance(x, CodebookSpec) for x in raw):
            return raw
        if all(isinstance(x, dict) for x in raw):
            return [CodebookSpec(**x) for x in raw]

    raw_str = str(raw)

    # First try literal-eval in case a dict-list was saved
    try:
        parsed = ast.literal_eval(raw_str)
        if isinstance(parsed, list) and all(isinstance(x, dict) for x in parsed):
            return [CodebookSpec(**x) for x in parsed]
    except Exception:
        pass

    # Fallback for repr list like:
    # [CodebookSpec(kind='narrow', N=8, K=None, M=None), ...]
    pattern = re.compile(
        r"CodebookSpec\(kind=(?P<q>['\"])(?P<kind>.+?)(?P=q),\s*"
        r"N=(?P<N>None|[-+]?\d+),\s*"
        r"K=(?P<K>None|[-+]?\d+),\s*"
        r"M=(?P<M>None|[-+]?\d+)\)"
    )

    specs: List[CodebookSpec] = []
    for match in pattern.finditer(raw_str):
        def none_or_int(x: str) -> Optional[int]:
            return None if x == "None" else int(x)

        specs.append(
            CodebookSpec(
                kind=match.group("kind"),
                N=none_or_int(match.group("N")),
                K=none_or_int(match.group("K")),
                M=none_or_int(match.group("M")),
            )
        )

    if specs:
        return specs

    raise ValueError(f"Could not parse codebook_specs from: {raw_str}")


def _read_params_sections(params_path: str) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], str]:
    if not os.path.exists(params_path):
        raise FileNotFoundError(f"Parameters file not found: {params_path}")

    current: Optional[str] = None
    common: Dict[str, Any] = {}
    dqn: Dict[str, Any] = {}
    ql: Dict[str, Any] = {}
    experiment_note = ""

    with open(params_path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue

            if line.startswith("Experiment note:"):
                experiment_note = line.split("Experiment note:", 1)[1].strip()
                continue

            if line == SECTION_COMMON:
                current = "common"
                continue
            if line == SECTION_DQN:
                current = "dqn"
                continue
            if line == SECTION_QL:
                current = "ql"
                continue

            if ":" not in line or current is None:
                continue

            key, value = line.split(":", 1)
            key = key.strip()
            value = _parse_scalar(value)

            if current == "common":
                common[key] = value
            elif current == "dqn":
                dqn[key] = value
            elif current == "ql":
                ql[key] = value

    if "codebook_specs" in common:
        common["codebook_specs"] = _parse_codebook_specs(common["codebook_specs"])

    return common, dqn, ql, experiment_note


def load_parameters_from_saved_config(example_root: str, mode: str = "auto") -> Parameters:
    paths = ExperimentPaths(root=example_root)

    candidate_files: List[str]
    if mode == "dqn":
        candidate_files = [
            paths.params_file("dqn"),
            paths.params_file("both"),
            paths.params_resume_training_file("dqn"),
            paths.params_resume_training_file("both"),
        ]
    elif mode == "ql":
        candidate_files = [
            paths.params_file("ql"),
            paths.params_file("both"),
            paths.params_resume_training_file("ql"),
            paths.params_resume_training_file("both"),
        ]
    else:
        candidate_files = [
            paths.params_file("both"),
            paths.params_file("dqn"),
            paths.params_file("ql"),
            paths.params_resume_training_file("both"),
            paths.params_resume_training_file("dqn"),
            paths.params_resume_training_file("ql"),
        ]

    params_path = next((p for p in candidate_files if os.path.exists(p)), None)
    if params_path is None:
        raise FileNotFoundError(
            f"No params file found under {paths.config_dir}. Checked: {candidate_files}"
        )

    common, dqn, ql, experiment_note = _read_params_sections(params_path)

    merged: Dict[str, Any] = {}
    merged.update(common)
    merged.update(dqn)
    merged.update(ql)

    parameters = Parameters(
        experiment_note=experiment_note,
        experiment_seed=merged.get("experiment_seed", 42),
        N_R=merged.get("N_R", 64),
        N_T=merged.get("N_T", 1),
        N_RIS=merged.get("N_RIS", 100),
        size_codebooks=merged.get("size_codebooks", [8, 14]),
        codebook_specs=common.get("codebook_specs"),
        SNR=merged.get("SNR", 10),
        snr_values=merged.get("snr_values", [0, 5, 10, 20]),
        type_channel=merged.get("type_channel", "half-spaced ULAs"),
        type_modulation=merged.get("type_modulation", "BPSK"),
        mean_noise=merged.get("mean_noise", 0),
        mean_channel=merged.get("mean_channel", 0),
        std_channel=merged.get("std_channel", []),
        sigma_alpha=merged.get("sigma_alpha", 0),
        hierarchical_noisy_measurement=merged.get("hierarchical_noisy_measurement", True),
        gamma=merged.get("gamma", 0.99),
        _greedy_mode=merged.get("_greedy_mode", False),
        learning_rate_init=merged.get("learning_rate_init", 5e-4),
        learning_rate_decay=merged.get("learning_rate_decay", 0.99),
        learning_rate_min=merged.get("learning_rate_min", 1e-4),
        epsilon_init=merged.get("epsilon_init", 1.0),
        epsilon_decay=merged.get("epsilon_decay", 0.999),
        epsilon_min=merged.get("epsilon_min", 0.01),
        delta_init=merged.get("delta_init", 1e-1),
        delta_decay=merged.get("delta_decay", 1),
        delta_min=merged.get("delta_min", 5e-2),
        params_list=dqn.get("params_list", [32, 64]),
        loss_fct=dqn.get("loss_fct", "mse"),
        batch_size=dqn.get("batch_size", 128),
        replay_buffer_memory_size=dqn.get("replay_buffer_memory_size", None),
        n_epochs=dqn.get("n_epochs", 10000),
        n_time_steps_dqn=dqn.get("n_time_steps", 200),
        n_channels_train_DQN=dqn.get("n_channels_train", 10),
        n_episodes=ql.get("n_episodes", 20),
        n_time_steps_ql=ql.get("n_time_steps", 100),
        n_channels_train_QL=ql.get("n_channels_train", 10),
        max_len_path=merged.get("max_len_path", 20),
        len_path=ql.get("len_path", merged.get("len_path", 10)),
        max_norm=dqn.get("max_norm", 1),
        do_gradient_clipping=dqn.get("do_gradient_clipping", True),
        tau=dqn.get("tau", 0.05),
        freq_update_target=dqn.get("freq_update_target", 1000),
        targetNet_update_method=dqn.get("targetNet_update_method", "soft"),
        Train_Deep_Q_Learning=False,
        Train_Q_Learning=False,
        saving_freq=merged.get("saving_freq", 1),
        test_freq=merged.get("test_freq", 1),
        continue_training=False,
        recover_checkpoint_path=example_root,
        enable_async_eval=False,
        async_eval_poll_seconds=dqn.get("async_eval_poll_seconds", 20),
        async_eval_device=dqn.get("async_eval_device", "cpu"),
        precision=ql.get("precision", 2),
        len_window_channel=merged.get("len_window_channel", 10),
        modification_channel=merged.get("modification_channel", 0),
        min_representatives_q_learning_train=ql.get("min_representatives_q_learning_train", 100),
        min_representatives_q_learning_test=ql.get("min_representatives_q_learning_test", 10),
    )

    print(f"[RETEST] Loaded params from: {params_path}")
    return parameters


def _find_available_q_policy(paths: ExperimentPaths):
    if not os.path.isdir(paths.q_policies_dir):
        return None

    ql_files = [
        f for f in os.listdir(paths.q_policies_dir)
        if f.startswith("policy_after_") and f.endswith("episodes.csv")
    ]
    if not ql_files:
        return None

    ql_files.sort(key=lambda x: int(x.split("_")[2].replace("episodes.csv", "")))
    latest = ql_files[-1]
    latest_epoch = int(latest.split("_")[2].replace("episodes.csv", ""))
    print(f"[RETEST] Loading one QL policy to enable test mode selection: {latest}")
    return load_Policy(paths=paths, episode=latest_epoch)


def _has_dqn_checkpoints(paths: ExperimentPaths) -> bool:
    if not os.path.isdir(paths.dqn_checkpoints_dir):
        return False
    return any(f.endswith("_eval.pth") for f in os.listdir(paths.dqn_checkpoints_dir))


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-run tests on an existing Example_N using saved config params.")
    parser.add_argument("example_root", help="Path like ./Data/Example_48")
    parser.add_argument("--mode", choices=["auto", "dqn", "ql", "both"], default="auto")
    parser.add_argument("--params-mode", choices=["auto", "dqn", "ql", "both"], default="auto")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    example_root = os.path.abspath(args.example_root)
    paths = ExperimentPaths(root=example_root)
    store = Store(paths)

    parameters = load_parameters_from_saved_config(
        example_root=example_root,
        mode=args.params_mode,
    )

    if args.seed is not None:
        parameters.experiment_seed = args.seed

    set_seed(parameters.experiment_seed)

    builder = ExperimentBuilder(parameters=parameters, store=store)
    context = builder.build(
        dataset_mode=DatasetMode.LOAD_GENERATED,
        noise_mode=NoiseMode.REUSE,
    )
    runner = Runner(context=context)

    requested_mode = args.mode
    q_policy = None
    dqn_agent = None
    dqn_policy = None

    ql_available = os.path.isdir(paths.q_policies_dir) and any(
        f.startswith("policy_after_") and f.endswith("episodes.csv")
        for f in os.listdir(paths.q_policies_dir)
    )
    dqn_available = _has_dqn_checkpoints(paths)

    if requested_mode == "auto":
        if ql_available and dqn_available:
            requested_mode = "both"
        elif dqn_available:
            requested_mode = "dqn"
        elif ql_available:
            requested_mode = "ql"
        else:
            raise FileNotFoundError(
                f"No QL policies or DQN checkpoints found under {example_root}"
            )

    print(f"[RETEST] Example root: {example_root}")
    print(f"[RETEST] Dataset file: {paths.dataset_pickle}")
    print(f"[RETEST] Noise file:   {paths.noise_csv}")
    print(f"[RETEST] Requested mode: {requested_mode}")

    if requested_mode in ("both", "ql"):
        q_policy = _find_available_q_policy(paths)
        if q_policy is None:
            raise FileNotFoundError(f"No QL policy found in {paths.q_policies_dir}")

    if requested_mode in ("both", "dqn"):
        if not dqn_available:
            raise FileNotFoundError(f"No DQN checkpoints found in {paths.dqn_checkpoints_dir}")

        dqn_agent = DeepQLearningAgent(
            parameters=parameters,
            environment=runner.env,
            paths=store.paths,
        )
        dqn_policy = dqn_agent.evaluation_q_network
        dqn_policy.eval()

    runner.run_testing(
        q_policy=q_policy,
        dqn_agent=dqn_agent,
        dqn_policy_net=dqn_policy,
    )

    print("[RETEST] Done.")


if __name__ == "__main__":
    main()
