use burn::{
    nn::{
        Dropout,
        DropoutConfig,
        LayerNorm,
        LayerNormConfig,
        Linear,
        LinearConfig,
        // attention::{MhaInput, MultiHeadAttention, MultiHeadAttentionConfig},
    },
    prelude::*,
    tensor::activation::relu,
};

use crate::multi_head_attention::{MultiHeadAttention, MultiHeadAttentionConfig};

#[derive(Debug, Config)]
pub struct EncoderLayerConfig {
    model_dimension: usize,
    head_count: usize,
    #[config(default = 0.10)]
    drop_out_prob: f64,
}
impl EncoderLayerConfig {
    pub fn init<B: Backend>(
        &Self {
            model_dimension,
            head_count,
            drop_out_prob,
        }: &Self,
        device: &B::Device,
    ) -> EncoderLayer<B> {
        EncoderLayer {
            self_attention: MultiHeadAttentionConfig::new(model_dimension, head_count).init(device),
            self_attention_norm: LayerNormConfig::new(model_dimension).init(device),
            linear: [
                LinearConfig::new(model_dimension, model_dimension).init(device),
                LinearConfig::new(model_dimension, model_dimension).init(device),
            ],
            linear_norm: LayerNormConfig::new(model_dimension).init(device),
            drop_out: DropoutConfig::new(drop_out_prob).init(),
        }
    }
}

#[derive(Debug, Module)]
pub struct EncoderLayer<B: Backend> {
    self_attention: MultiHeadAttention<B>,
    self_attention_norm: LayerNorm<B>,
    linear: [Linear<B>; 2],
    linear_norm: LayerNorm<B>,
    drop_out: Dropout,
}
impl<B: Backend> EncoderLayer<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        // attention
        let query = x.clone();
        let key = x.clone();
        let value = x.clone();
        let mask = None;
        let self_attention_output = self.self_attention.forward(query, key, value, mask);
        let self_attention_output = self.drop_out.forward(self_attention_output);
        let self_attention_output = self
            .self_attention_norm
            .forward(x.clone() + self_attention_output);

        // feed forward
        let linear_output = self.linear[0].forward(self_attention_output.clone());
        let linear_output = relu(linear_output);
        let linear_output = self.linear[1].forward(linear_output);
        let linear_output = self.drop_out.forward(linear_output);
        self.linear_norm.forward(x + linear_output)
    }
}
