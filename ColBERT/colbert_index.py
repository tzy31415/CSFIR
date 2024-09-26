from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert import Indexer
import argparse

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--nranks", type=int, default=1)
    parser.add_argument("--experiment_name", type=str)
    parser.add_argument("--root", type=str)
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument("--index_name", type=str)
    parser.add_argument("--collection_file", type=str)
    args = parser.parse_args()

    
    with Run().context(RunConfig(nranks=args.nranks, experiment=args.experiment_name)):

        config = ColBERTConfig(
            nbits=2,
            root=args.root,
        )
        indexer = Indexer(checkpoint=args.checkpoint, config=config)
        indexer.index(name=args.index_name,collection=args.collection_file)
        
