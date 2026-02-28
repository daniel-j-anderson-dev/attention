use burn::{
    nn::{Dropout, LayerNorm, Linear, LinearConfig},
    prelude::*,
    tensor::activation::relu,
};

#[derive(Debug, Config)]
pub struct MultiHeadAttentionConfig {
    model_dimension: usize,
    head_dimension: usize,
    attention_head_count: usize,
}
impl MultiHeadAttentionConfig {
    pub fn init<B: Backend>(self, device: &B::Device) -> MultiHeadAttention<B> {
        let MultiHeadAttentionConfig {
            model_dimension,
            head_dimension,
            attention_head_count,
        } = self;

        let linear_no_bias = || {
            LinearConfig::new(model_dimension, model_dimension)
                .with_bias(false)
                .init::<B>(device)
        };

        let linear = || LinearConfig::new(model_dimension, model_dimension).init::<B>(device);

        MultiHeadAttention {
            model_dimension,
            head_dimension,
            attention_head_count,

            query: linear_no_bias(),
            key: linear_no_bias(),
            value: linear_no_bias(),

            dense: linear(),
            output: linear(),
        }
    }
}

#[derive(Debug, Module)]
pub struct MultiHeadAttention<B: Backend> {
    model_dimension: usize,
    head_dimension: usize,
    attention_head_count: usize,

    query: Linear<B>,
    key: Linear<B>,
    value: Linear<B>,

    dense: Linear<B>,
    output: Linear<B>,
}
impl<B: Backend> MultiHeadAttention<B> {
    pub fn forward<const D: usize>(
        &self,
        query: Tensor<B, D>,
        key: Tensor<B, D>,
        value: Tensor<B, D>,
        mask: Option<Tensor<B, D>>,
    ) {
        todo!()
    }
}
