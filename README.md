# Installation for Inference in HF
To perform inference from the Huggingface converted version of the model GPT-2B-001_bf16_tp1.nemo, you might need to create the following environment: 

- `python3.9 -m venv env_infer`
- `source env_infer/bin/activate`
- `pip install -r requirements.txt`
- `pip install torch==2.0.1+cu118 --extra-index-url https://download.pytorch.org/whl/cu118/`
- `pip install git+https://github.com/ertkonuk/transformers.git`
- `pip install flash-attn==2.0.5`

We needed to use flash-attn==2.0.5 due to CUDA driver issues and this requires to modify your [transformers github repository](https://github.com/ertkonuk/transformers/tree/main). Specifically, we had to modify two files:
- In the file `.../transformers/models/modeling_nvgpt.py`, we needed to change all names:
flash_attn_unpadded_func > flash_attn_varlen_func
Justification: https://pypi.org/project/flash-attn/2.3.3/.
- In `.../torch/nn/modules/normalization.py` we needed to modify the class LayerNorm by:
```
class LayerNorm(Module):
    r"""Applies Layer Normalization over a mini-batch of inputs as described in
    the paper `Layer Normalization <https://arxiv.org/abs/1607.06450>`__

    .. math::
        y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    The mean and standard-deviation are calculated over the last `D` dimensions, where `D`
    is the dimension of :attr:`normalized_shape`. For example, if :attr:`normalized_shape`
    is ``(3, 5)`` (a 2-dimensional shape), the mean and standard-deviation are computed over
    the last 2 dimensions of the input (i.e. ``input.mean((-2, -1))``).
    :math:`\gamma` and :math:`\beta` are learnable affine transform parameters of
    :attr:`normalized_shape` if :attr:`elementwise_affine` is ``True``.
    The standard-deviation is calculated via the biased estimator, equivalent to
    `torch.var(input, unbiased=False)`.

    .. note::
        Unlike Batch Normalization and Instance Normalization, which applies
        scalar scale and bias for each entire channel/plane with the
        :attr:`affine` option, Layer Normalization applies per-element scale and
        bias with :attr:`elementwise_affine`.

    This layer uses statistics computed from input data in both training and
    evaluation modes.

    Args:
        normalized_shape (int or list or torch.Size): input shape from an expected input
            of size

            .. math::
                [* \times \text{normalized\_shape}[0] \times \text{normalized\_shape}[1]
                    \times \ldots \times \text{normalized\_shape}[-1]]

            If a single integer is used, it is treated as a singleton list, and this module will
            normalize over the last dimension which is expected to be of that specific size.
        eps: a value added to the denominator for numerical stability. Default: 1e-5
        elementwise_affine: a boolean value that when set to ``True``, this module
            has learnable per-element affine parameters initialized to ones (for weights)
            and zeros (for biases). Default: ``True``.

    Attributes:
        weight: the learnable weights of the module of shape
            :math:`\text{normalized\_shape}` when :attr:`elementwise_affine` is set to ``True``.
            The values are initialized to 1.
        bias:   the learnable bias of the module of shape
                :math:`\text{normalized\_shape}` when :attr:`elementwise_affine` is set to ``True``.
                The values are initialized to 0.

    Shape:
        - Input: :math:`(N, *)`
        - Output: :math:`(N, *)` (same shape as input)

    Examples::

        >>> # NLP Example
        >>> batch, sentence_length, embedding_dim = 20, 5, 10
        >>> embedding = torch.randn(batch, sentence_length, embedding_dim)
        >>> layer_norm = nn.LayerNorm(embedding_dim)
        >>> # Activate module
        >>> layer_norm(embedding)
        >>>
        >>> # Image Example
        >>> N, C, H, W = 20, 5, 10, 10
        >>> input = torch.randn(N, C, H, W)
        >>> # Normalize over the last three dimensions (i.e. the channel and spatial dimensions)
        >>> # as shown in the image below
        >>> layer_norm = nn.LayerNorm([C, H, W])
        >>> output = layer_norm(input)

    .. image:: ../_static/img/nn/layer_norm.jpg
        :scale: 50 %

    """
    __constants__ = ['normalized_shape', 'eps', 'elementwise_affine']
    normalized_shape: Tuple[int, ...]
    eps: float
    elementwise_affine: bool

    def __init__(self, normalized_shape: _shape_t, eps: float = 1e-5, elementwise_affine: bool = True,
                 bias: bool = True, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            # mypy error: incompatible types in assignment
            normalized_shape = (normalized_shape,)  # type: ignore[assignment]
        self.normalized_shape = tuple(normalized_shape)  # type: ignore[arg-type]
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = Parameter(torch.empty(self.normalized_shape, **factory_kwargs))
            if bias:
                self.bias = Parameter(torch.empty(self.normalized_shape, **factory_kwargs))
            self.bias = Parameter(torch.empty(self.normalized_shape, **factory_kwargs))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.elementwise_affine:
            init.ones_(self.weight)
            if self.bias is not None:
                init.zeros_(self.bias)
            # init.zeros_(self.bias)

    def forward(self, input: Tensor) -> Tensor:
        return F.layer_norm(
            input, self.normalized_shape, self.weight, self.bias, self.eps)

    def extra_repr(self) -> str:
        return '{normalized_shape}, eps={eps}, ' \
            'elementwise_affine={elementwise_affine}'.format(**self.__dict__)
```

## Steps to reproduce our results
1. Download the model `GPT-2B-001_bf16_tp1.nemo` (https://huggingface.co/nvidia/GPT-2B-001/blob/main/GPT-2B-001_bf16_tp1.nemo) and perform inference by using:
`python load_and_infer_nemo.py gpt_model_file=./GPT-2B-001_bf16_tp1.nemo trainer.precision=bf16 server=True tensor_model_parallel_size=1 inference.greedy=True trainer.devices=1 inference.compute_logprob=True prompts=["Life is like a","How are you?"]`
(code extracted from [/megatron_gpt_eval.py](https://github.com/NVIDIA/NeMo/blob/main/examples/nlp/language_modeling/megatron_gpt_eval.py))
This code will save in `./example_pred` the logprob distribution assocaited with the prompts given.
The installation procedure for this is more complex to explain, but we suppose that you already have an inference pipeline for this model. If you have problems, please let us know.
2. Then, we use the code in `convert_nemo_to_hf.py` to change the format from .nemo to huggingface. This step is equivalent to what you provided.
3. Finally, we load the model using `inference_hf.py` and find that the results given when we use the huggingface version and the .nemo version are different. Note that even when setting the parameters to bfloat16 precision, the precision given seems to be 8-bit.

The results from this code are:
Logits from HF:
>>> logits_hf
tensor([[[-28.5000, -28.5000, -28.5000,  ..., -28.5000, -28.5000, -28.5000],
         [-28.3750, -28.3750, -28.3750,  ..., -28.3750, -28.3750, -28.3750],
         [-26.2500, -26.2500, -26.2500,  ..., -26.2500, -26.2500, -26.2500],
         [-20.3750, -20.3750, -20.3750,  ..., -20.3750, -20.3750, -20.3750]]],
       device='cuda:0', dtype=torch.bfloat16, grad_fn=<TransposeBackward0>)

Logits from .nemo:
>>> logits_nemo
tensor([[-17.4963, -17.4963, -17.4963,  ..., -17.4963, -17.4963, -17.4963],
        [-19.2986, -19.2986, -19.2986,  ..., -19.2986, -19.2986, -19.2986],
        [-20.0501, -20.0501, -20.0501,  ..., -20.0501, -20.0501, -20.0501],
        [-19.2978, -19.2978, -19.2978,  ..., -19.2978, -19.2978, -19.2978]])