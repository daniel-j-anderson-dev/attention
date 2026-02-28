use burn::{
    nn::{Linear, LinearConfig},
    prelude::*,
    tensor::activation::{relu, softmax},
};

#[derive(Debug, Config)]
pub struct MultiHeadAttentionConfig {
    model_dimension: usize,
    head_dimension: usize,
    attention_head_count: usize,
}
impl MultiHeadAttentionConfig {
    pub fn init<B: Backend>(
        &Self {
            model_dimension,
            head_dimension,
            attention_head_count,
        }: &Self,
        device: &B::Device,
    ) -> MultiHeadAttention<B> {
        MultiHeadAttention {
            model_dimension,
            head_dimension,
            attention_head_count,

            query: LinearConfig::new(model_dimension, model_dimension)
                .with_bias(false)
                .init(device),
            key: LinearConfig::new(model_dimension, model_dimension)
                .with_bias(false)
                .init(device),
            value: LinearConfig::new(model_dimension, model_dimension)
                .with_bias(false)
                .init(device),

            dense: LinearConfig::new(model_dimension, model_dimension).init(device),
            output: LinearConfig::new(model_dimension, model_dimension).init(device),
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
    fn scaled_dot_product_attention(
        &self,
        query: Tensor<B, 4>,
        key: Tensor<B, 4>,
        value: Tensor<B, 4>,
        mask: Option<Tensor<B, 4, Bool>>,
    ) -> Tensor<B, 4> {
        let key_t = key.permute([0, 1, 3, 2]);
        let sqrt_head_dimension = (self.head_dimension as f64).sqrt();
        let mut scores = query.matmul(key_t) / sqrt_head_dimension;
        if let Some(mask) = mask {
            scores = scores.mask_fill(mask, f32::NEG_INFINITY);
        }
        let scores = softmax(scores, 3);
        scores.matmul(value)
    }

    pub fn forward(
        &self,
        query_input: Tensor<B, 3>,
        key_input: Tensor<B, 3>,
        value_input: Tensor<B, 3>,
        mask: Option<Tensor<B, 4, Bool>>,
    ) -> Tensor<B, 3> {
        let [batch_size, input_length, _input_dimension] = query_input.dims();

        // initial weight projections
        let query_output = self.query.forward(query_input);
        let key_output = self.key.forward(key_input);
        let value_output = self.value.forward(value_input);

        // reshape
        // from: (batch_size, input_length, input_dimension)
        // to:   (batch_size, input_length, attention_head_count, head_dimension)
        let new_shape = [
            batch_size,
            input_length,
            self.attention_head_count,
            self.head_dimension,
        ];
        let query_output = query_output.reshape(new_shape);
        let key_output = key_output.reshape(new_shape);
        let value_output = value_output.reshape(new_shape);

        // reshape. swap dimensions 1 (input_length) and 2 (attention_head_count)
        // from: (batch_size, input_length, attention_head_count, head_dimension)
        // to:   (batch_size, attention_head_count, input_length, head_dimension)
        let new_dimension_order = [0, 2, 1, 3];
        let query_output = query_output.permute(new_dimension_order);
        let key_output = key_output.permute(new_dimension_order);
        let value_output = value_output.permute(new_dimension_order);

        // scaled dot product attention
        let scores =
            self.scaled_dot_product_attention(query_output, key_output, value_output.clone(), mask);

        let attention_weights = softmax(scores, 3);
        let context_values = attention_weights.matmul(value_output);

        // final output layer
        let context_values = context_values.reshape([
            batch_size,
            input_length,
            self.attention_head_count * self.head_dimension,
        ]);
        let dense_output = self.dense.forward(context_values);
        let dense_output = relu(dense_output);
        let dense_output = self.output.forward(dense_output);

        dense_output
    }
}
