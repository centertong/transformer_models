# Transformers Test

Linear Complexity Transformers Test using KLUE dataset

## Models

* bert: BERT (From huggingface transformers library)
* performer: Performer (From <https://github.com/lucidrains/performer-pytorch>)
* rfa: Random Feature Attention [Only 'arccos' mode]
* lite: Lite Transformer
* aft: Attention Free Transformer
* luna: Linear Unified Nested Attention
* fastformer: Fastformer
* scatterbrain: ScatterBrain
* abc: ATTENTION WITH BOUNDED-MEMORY CONTROL
* scaling: Scaling Transformers [work in progress]

## Usages

* Running klue dataset

```bash
python run/train_klue.py --config {model_config} --tokenizer {tokenizer_file} \ 
--type {dataset_type} --epochs {epoch_num} --batch_size {batchsize} --name {log_name}
```

* Example

```batsh
python run/train_klue.py --config configs/model/bert.json --name bert \
 --tokenizer vocab/bert2/tokenizer.json --type tc --epoch 10 --batch_size 12 
```

## Tests

### KLUE Dataset (STS, NLI, TC)

During 10 epoch training, select best performance for validation set.  
tokenizer was created with kowiki dataset by huggingface tokenizers library.

|Model|TC(MacroF1)|STS(Accuracy)|NLI(Pearson)|
|-----|---|---|---|
|Transformer Encoder|0.776|0.416|0.382|
|Performer|0.750| - |0.340|
|rfa|0.774|0.416|0.377|
|lite|0.769|0.388|0.402|
|aft|0.732| - |0.374|
|luna|0.562|0.356|0.397|
|fastformer|0.775|0.363|0.422|
|scatterbrain|0.447|0.377|0.345|
|abc|0.768| - |0.403|
|scaling|0.733| - | - |

## Reference

1. [hugginface transformers][1]
2. [hugginface tokenizers][2]
3. [performer-pytorch][3]
4. [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding][4]
5. [Rethinking Attention with Performers][5]
6. [Random Feature Attention][6]
7. [Lite Transformer with Long-Short Range Attention][7]
8. [An Attention Free Transformer][8]
9. [Luna: Linear Unified Nested Attention][9]
10. [Fastformer: Additive Attention Can Be All You Need][10]
11. [Scatterbrain: Unifying Sparse and Low-rank Attention Approximation][11]
12. [ABC: ATTENTION WITH BOUNDED-MEMORY CONTROL][12]
13. [Sparse is Enough in Scaling Transformers][13]

[1]: https://github.com/huggingface/transformers
[2]: https://github.com/huggingface/tokenizers
[3]: https://github.com/lucidrains/performer-pytorch
[4]: https://arxiv.org/abs/1810.04805
[5]: https://arxiv.org/abs/2009.14794
[6]: https://arxiv.org/abs/2103.02143
[7]: https://arxiv.org/abs/2004.11886
[8]: https://arxiv.org/abs/2105.14103
[9]: https://arxiv.org/abs/2106.01540
[10]: https://arxiv.org/abs/2108.09084
[11]: https://arxiv.org/abs/2110.15343
[12]: https://arxiv.org/abs/2110.02488
[13]: https://arxiv.org/abs/2111.12763
