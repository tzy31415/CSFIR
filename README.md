# CSFIR

This is the implementation of the paper 'CSFIR: Leveraging Code-Specific Features to Augment Information Retrieval'.

## Dataset

We utilize the dataset provided by [CodeSearchNet](https://github.com/github/CodeSearchNet). From this dataset, we randomly sampled 500 examples to simulate a low-resource scenario for our initial experiments.

## Model

Our implementation of ColBERT uses [BERT](https://huggingface.co/google-bert/bert-base-uncased) ,as the backbone model. This pre-trained model serves as the foundation for the contextual encoding of both queries and documents, which enables efficient and effective retrieval in our system.

## Resources

The trained model, index, and retrieval results are available at the following link: [Google Drive](https://drive.google.com/drive/folders/1bH9NEywDyaTw3tZdSEwqeBS6dINX1LcH?usp=drive_link) or [Baidu Drive](https://pan.baidu.com/s/1Jii50Ae43ZmI9G9UwIBO6Q&pwd=igd3) .

## Tuning LLM and Generating Pseudo Query

In this project, we use [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) to fine-tune the large language model (LLM) for generating pseudo queries. The specific steps are as follows:

First, organize your dataset into the required format for LLaMA-Factory, mapping code snippets to their corresping natural language descriptions. Save this as the appropriate training data file, or you can use the dataset files I have already prepared.

Use LLaMA-Factory to fine-tune a pre-trained LLM for query generation. Run the following command to initiate the fine-tuning process:

```bash
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train examples/SFT.yaml
```

Once the model has been fine-tuned, you can generate pseudo queries using the trained model. Execute the following command to generate the pseudo queries:

```bash
CUDA_VISIBLE_DEVICES=0 llamafactory-cli generation examples/generation.yaml
```

By following these steps, you can generate high-quality pseudo queries for downstream retrieval tasks. The specific fine-tuning parameters and query generation processes can be adjusted based on the task requirements.

### Tips for Data Preparation

After generating the pseudo queries, it's essential to organize the data into the following formats for further use:

* `collection.tsv`: Contains the corpus (code snippets).
* `queries.tsv`: Contains the pseudo queries.
* `triples.json`: Maps each query to its corresponding relevant and non-relevant documents.

These files should align with the structure found in `data/python_test`.

 **Important Note** : When working with TSV files, there can sometimes be formatting issues, particularly when storing and handling files with certain characters, which may lead to failures in building Abstract Syntax Trees (AST). To address this, we introduced a `python_ast` file to assist in the AST building process, ensuring that code snippets are properly parsed and indexed. Make sure to include and use this file in the pipeline to avoid AST-related issues.

## ColBERT Related Operations

In this section, we describe how to use a ColBERT model enhanced with Graph Convolutional Networks (GCN) to perform retrieval over a code corpus. The workflow is as follows:

**Train ColBERT with Pseudo Queries** : First, use the pseudo query data to train the ColBERT model.

```bash
python colbert_train.py \
--nranks 1 \
--experiment_name experiment_name \
--batch_size batch_size \
--max_steps max_steps \
--root root \
--checkpoint checkpoint \
--triples_file triples_file \
--queries_file queries_file \
--collection_file collection_file
```

**Indexing with FAISS and ColBERT** : After training, use FAISS in combination with ColBERT to build an efficient index for the code corpus. This index will allow fast and accurate retrieval. Run the following command to create the index:

```bash
python colbert_index.py \
--nranks 1 \
--experiment_name experiment_name \
--root root \
--checkpoint checkpoint_file \
--index_name index_name \
--collection_file collection_file
```

**Perform Retrieval and Evaluate** : Once the index is built, perform retrieval on the code corpus using the trained ColBERT model. After retrieving the results, evaluate the final performance of the model to check how well it performs on the task. Use the following command to run the retrieval  process:

```bash
python colbert_retrieval.py \
--nranks 1 \
--experiment_name experiment_name \
--root root \
--checkpoint checkpoint \
--index_name index_name \
--query_file query_file \
--rank_file rank_file
```

**Evaluation the performance:**

```bash
python -m utility.evaluate.msmarco_passages --ranking rank_file --qrels qrels_file
```

By following these steps, you can leverage the combination of ColBERT and GCN to efficiently retrieve relevant code snippets and assess the model's overall effectiveness on the code retrieval task.

Finally, an explanation of the parameters:

* `nrank`: The number of GPUs used for the process.
* `experiment_name`: The name of the experiment, used to create the directory for storing the model.
* `root`: The root directory path for storing data and models.
* `checkpoint`: The path to the pre-trained BERT model.
* `queries_file`: The file containing the queries.
* `collection_file`: The file containing the corpus (code snippets).
* `triples_file`: The file mapping queries to corresponding corpus entries.
* `index_name`: The name of the index, which determines where the index files will be saved.
* `rank_file`: The file where the retrieval results will be stored.
* `qrels_file`: The file containing the ground truth retrieval results for evaluation.

## Acknowledgements

We would like to express our gratitude to the developers of the following repositories, which significantly inspired and contributed to the development of our code:

* [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory): This repository provided valuable insights into large language model implementations, which we utilized to enhance our own model structure and performance.
* [ColBERT](https://github.com/stanford-futuredata/ColBERT): The retrieval mechanisms and ranking algorithms in this repository were instrumental in shaping our approach to efficient document retrieval and scoring.

Their contributions to the open-source community have been indispensable, and we are deeply appreciative of their efforts.
