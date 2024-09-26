from colbert.data import Queries
from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert import Searcher
import argparse
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--nranks", type=int, default=1)
    parser.add_argument("--experiment_name", type=str)
    parser.add_argument("--root", type=str)
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument("--index_name", type=str)
    parser.add_argument("--query_file", type=str)
    parser.add_argument("--rank_file", type=str)

    args = parser.parse_args()
    
    with Run().context(RunConfig(nranks=args.nranks, experiment=args.experiment_name)):
        config = ColBERTConfig(
            root=args.root,
            checkpoint=args.checkpoint    
            )
        searcher = Searcher(index=args.index_name, config=config)
        queries = Queries(args.query_file)
        ranking = searcher.search_all(queries, k=100)
        ranking.save(args.rank_file)
        