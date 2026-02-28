use burn::{
    nn::{
        Dropout, LayerNorm, Linear,
        attention::{MhaInput, MultiHeadAttention},
    },
    prelude::*,
    tensor::activation::relu,
};

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
        let [query, key, value] = [(); _].map(|_| x.clone());
        let self_attention_input = MhaInput::new(query, key, value);
        let self_attention_output = self.self_attention.forward(self_attention_input);
        let self_attention_output = self.drop_out.forward(self_attention_output.context);
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
