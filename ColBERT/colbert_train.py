from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert import Trainer
import argparse





if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--nranks", type=int, default=1)
    parser.add_argument("--experiment_name", type=str)
    parser.add_argument("--batch_size", type=int,default=64)
    parser.add_argument("--max_steps", type=int, default=10000)
    parser.add_argument("--root", type=str)
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument("--triples_file", type=str)
    parser.add_argument("--queries_file", type=str)
    parser.add_argument("--collection_file", type=str)
    parser.add_argument("--python_ast_file", type=str,default='data/python_train/python_ast.json')

    args = parser.parse_args()
    with Run().context(RunConfig(nranks=args.nranks, experiment=args.experiment_name)):

        config = ColBERTConfig(
            bsize=args.batch_size,
            root=args.root,
            checkpoint=args.checkpoint,
            maxsteps = args.max_steps
        )
        trainer = Trainer(
            triples=args.triples_file,
            queries=args.queries_file,
            collection=args.collection_file,
            config=config,
            python_ast_file=args.python_ast_file
        )
        # here

        checkpoint_path = trainer.train( checkpoint=args.checkpoint)

        print(f"Saved checkpoint to {checkpoint_path}...")