import os
import sys
import argparse
from utils import import_custom_class


def main():

    parser = argparse.ArgumentParser(
        description="Arguments for the main train program."
    )
    parser.add_argument('--config_file', type=str, required=True, help='Path for the config file')
    parser.add_argument('--runner_class_path', type=str, default="runner/ge_trainer.py")
    parser.add_argument('--runner_class', type=str, default="Trainer")
    parser.add_argument('--mode', type=str, default="train")
    parser.add_argument('--resume', action='store_true', help='Resume training from the latest checkpoint in output_dir')
    parser.add_argument('--sub_folder', type=str, default=None, help='Subdirectory under output_dir to save/load checkpoints (useful for continuing a previous run)')
    parser.add_argument('--checkpoint_path', type=str, default=None, help='Path to trained checkpoint, used in inference stage only')
    parser.add_argument('--n_validation', type=int, default=1, help='num of samples to predict, used in inference stage only')
    parser.add_argument('--n_chunk_action', type=int, default=1, help='num of action chunks to predict, used in action inference stage only')
    parser.add_argument('--output_path', type=str, default=None, help='Path to save outputs, used in inference stage only')
    parser.add_argument('--domain_name', type=str, default="agibotworld", help='Domain name of the validation dataset, used in inference stage only')
    parser.add_argument('--statistics_domain', type=str, default=None, help='Domain name for statistics lookup, overrides domain_name if set')
    parser.add_argument('--tasks_per_run', type=int, default=None, help='Number of distinct tasks to evaluate during inference')
    parser.add_argument('--episodes_per_task', type=int, default=None, help='Number of episodes to sample per task during inference')
    parser.add_argument('--rollout_steps', type=int, default=0, help='When >0, perform sequential rollout for the specified number of steps instead of a single open-loop chunk.')
    parser.add_argument('--occlude_view', type=str, default=None, help='Name (or index) of the camera view to occlude during inference.')
    parser.add_argument('--occlude_start', type=int, default=None, help='Start timestep (inclusive) for occlusion. Defaults to 0 when occlude_view is set.')
    parser.add_argument('--occlude_end', type=int, default=None, help='End timestep (exclusive) for occlusion. Defaults to the total clip length when occlude_view is set.')

    args = parser.parse_args()
    
    if args.mode == "infer":
        print(
            f"[DEBUG main] Inference parameters: n_validation={args.n_validation}, "
            f"n_chunk_action={args.n_chunk_action}, domain_name={args.domain_name}, "
            f"tasks_per_run={args.tasks_per_run}, episodes_per_task={args.episodes_per_task}, "
            f"rollout_steps={args.rollout_steps}"
        )
    
    Runner = import_custom_class(
        args.runner_class, args.runner_class_path, 
    )
    

    if args.mode == "train":
        ### Trainer
        runner = Runner(args.config_file, resume=args.resume, sub_folder=args.sub_folder)
        runner.prepare_dataset()
        runner.prepare_models()
        runner.prepare_trainable_parameters()
        runner.prepare_optimizer()
        runner.prepare_for_training()
        runner.prepare_trackers()
        runner.train()

    elif args.mode == "infer":
        ### Inference
        runner = Runner(args.config_file, output_dir=args.output_path)

        episodes_per_task = args.episodes_per_task if args.episodes_per_task and args.episodes_per_task > 0 else 1
        tasks_per_run = args.tasks_per_run if args.tasks_per_run and args.tasks_per_run > 0 else None
        if tasks_per_run is not None:
            args.n_validation = episodes_per_task * tasks_per_run
        setattr(runner.args, "episodes_per_task", episodes_per_task)
        if tasks_per_run is not None:
            setattr(runner.args, "tasks_per_run", tasks_per_run)

        if args.checkpoint_path is not None:
            runner.args.load_weights = True
            runner.args.load_diffusion_model_weights = True
            runner.args.diffusion_model['model_path'] = args.checkpoint_path
        runner.prepare_val_dataset()
        runner.prepare_models()

        if args.occlude_view is not None:
            setattr(runner.args, "occlude_view", args.occlude_view)
        if args.occlude_start is not None:
            setattr(runner.args, "occlude_start", max(args.occlude_start, 0))
        if args.occlude_end is not None:
            setattr(runner.args, "occlude_end", max(args.occlude_end, 0))
        if args.rollout_steps and args.rollout_steps > 0:
            runner.rollout(
                rollout_steps=args.rollout_steps,
                n_validation=args.n_validation,
                domain_name=args.domain_name,
                tasks_per_run=tasks_per_run,
                episodes_per_task=episodes_per_task,
                statistics_domain=args.statistics_domain,
            )
        else:
            runner.infer(
                n_chunk_action=args.n_chunk_action,
                n_validation=args.n_validation,
                domain_name=args.domain_name,
                tasks_per_run=tasks_per_run,
                episodes_per_task=episodes_per_task,
                statistics_domain=args.statistics_domain,
            )

    else:
        raise NotImplementedError



if __name__ == "__main__":
  
    main()
    