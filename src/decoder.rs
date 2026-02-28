use burn::{
    nn::{
        Dropout, LayerNorm, Linear,
        attention::{MhaInput, MultiHeadAttention},
    },
    prelude::*,
    tensor::activation::relu,
};

#[derive(Debug, Module)]
pub struct DecoderLayer<B: Backend> {
    self_attention: MultiHeadAttention<B>,
    self_attention_norm: LayerNorm<B>,

    cross_attention: MultiHeadAttention<B>,
    cross_attention_norm: LayerNorm<B>,

    linear: [Linear<B>; 2],
    linear_norm: LayerNorm<B>,

    drop_out: Dropout,
}
