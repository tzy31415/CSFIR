import torch

from colbert.infra.run import Run
from colbert.utils.utils import print_message, batch


class CollectionEncoder():
    def __init__(self, config, checkpoint):
        self.config = config
        self.checkpoint = checkpoint
        self.use_gpu = self.config.total_visible_gpus > 0

    def encode_passages(self, passages,sampled_pids):
        Run().print(f"#> Encoding {len(passages)} passages..")
        # print('*'*100)
        # print(len(passages))
        # print(type(sampled_pids))
        # print(len(sampled_pids))
        # print(sampled_pids[0])
        # print("&"*100)
        if len(passages) == 0:
            return None, None
        
        
        result = []
        print(len(sampled_pids))
        print(len(passages))
        for i in range(len(sampled_pids)):
            result.append([passages[i], sampled_pids[i]])

        with torch.inference_mode():
            embs, doclens = [], []

            # Batch here to avoid OOM from storing intermediate embeddings on GPU.
            # Storing on the GPU helps with speed of masking, etc.
            # But ideally this batching happens internally inside docFromText.
            for passages_batch in batch(result, self.config.index_bsize * 50):
                passages_list = []
                sampled_pids = []
                for i in passages_batch:
                    
                    passages_list.append(i[0])
                    sampled_pids.append(i[1])
                embs_, doclens_ = self.checkpoint.docFromText(passages_list, bsize=self.config.index_bsize,
                                                              keep_dims='flatten', showprogress=(not self.use_gpu),sampled_pids = sampled_pids)
                embs.append(embs_)
                doclens.extend(doclens_)

            embs = torch.cat(embs)

            # embs, doclens = self.checkpoint.docFromText(passages, bsize=self.config.index_bsize,
            #                                                   keep_dims='flatten', showprogress=(self.config.rank < 1))

        # with torch.inference_mode():
        #     embs = self.checkpoint.docFromText(passages, bsize=self.config.index_bsize,
        #                                        keep_dims=False, showprogress=(self.config.rank < 1))
        #     assert type(embs) is list
        #     assert len(embs) == len(passages)

        #     doclens = [d.size(0) for d in embs]
        #     embs = torch.cat(embs)

        return embs, doclens
