use crate::multi_head_attention::{MultiHeadAttention, MultiHeadAttentionConfig};

use burn::{
    nn::{Dropout, LayerNorm, Linear},
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
impl<B: Backend> DecoderLayer<B> {
    pub fn forward(
        &self,
        x: Tensor<B, 3>, // f
        encoder_out: Tensor<B, 3>,
    ) -> Tensor<B, 3> {
        // self attention
        let query = x.clone();
        let key = x.clone();
        let value = x.clone();
        let mask = None;
        let self_attention_output = self.self_attention.forward(query, key, value, mask);
        let self_attention_output = self.drop_out.forward(self_attention_output);
        let self_attention_output = self.self_attention_norm.forward(x + self_attention_output);

        // cross attention
        let query = self_attention_output.clone();
        let key = encoder_out.clone();
        let value = encoder_out;
        let mask = None;
        let cross_attention_output = self.cross_attention.forward(query, key, value, mask);
        let cross_attention_output = self.drop_out.forward(cross_attention_output);
        let cross_attention_output = self.cross_attention_norm.forward(cross_attention_output);

        // feed forward
        let linear_output = self.linear[0].forward(cross_attention_output);
        let linear_output = relu(linear_output);
        let linear_output = self.linear[1].forward(linear_output);
        let linear_output = self.drop_out.forward(linear_output);
        let linear_output = self.linear_norm.forward(linear_output);
        linear_output
    }
}
