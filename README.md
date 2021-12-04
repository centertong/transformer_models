# Transformers Test
Linear Complexity Transformers Test using KLUE dataset

## Models
* BERT (From huggingface transformers library)
* Performer (From https://github.com/lucidrains/performer-pytorch)
* RFA (Random Feature Attention) [Only 'arccos' mode]
* Lite Transformer
* Attention Free Transformer
* Luna: Linear Unified Nested Attention
* Fastformer [work in progress]
* ScatterBrain [work in progress]

---

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