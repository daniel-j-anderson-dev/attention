use burn::{
    nn::{Linear, LinearConfig},
    prelude::*,
    tensor::activation::softmax,
};

#[derive(Debug, Config)]
pub struct MultiHeadAttentionConfig {
    model_dimension: usize,
    head_count: usize,
}
impl MultiHeadAttentionConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> MultiHeadAttention<B> {
        let &Self {
            model_dimension,
            head_count,
        } = self;
        MultiHeadAttention {
            model_dimension,
            head_count,

            query: LinearConfig::new(model_dimension, model_dimension)
                .with_bias(false)
                .init(device),
            key: LinearConfig::new(model_dimension, model_dimension)
                .with_bias(false)
                .init(device),
            value: LinearConfig::new(model_dimension, model_dimension)
                .with_bias(false)
                .init(device),

            output_projection: LinearConfig::new(model_dimension, model_dimension).init(device),
        }
    }
}

#[derive(Debug, Module)]
pub struct MultiHeadAttention<B: Backend> {
    model_dimension: usize,
    head_count: usize,

    query: Linear<B>,
    key: Linear<B>,
    value: Linear<B>,

    output_projection: Linear<B>,
}
impl<B: Backend> MultiHeadAttention<B> {
    fn head_dimension(&self) -> usize {
        self.model_dimension / self.head_count
    }

    fn scaled_dot_product_attention(
        &self, // MultiHeadAttention
        query: Tensor<B, 4>,
        key: Tensor<B, 4>,
        value: Tensor<B, 4>,
        mask: Option<Tensor<B, 4, Bool>>,
    ) -> Tensor<B, 4> {
        let d_k = self.head_dimension() as f64;
        let key_t = key.permute([0, 1, 3, 2]);

        // Q * K^T
        let scores = query.matmul(key_t);

        // scale
        let mut scores = scores / d_k.sqrt();

        // masking
        if let Some(mask) = mask {
            scores = scores.mask_fill(mask, 1e-9);
        }

        let weights = softmax(scores, 3);
        weights.matmul(value)
    }

    /// - Reshapes `x`:
    ///   - from: `[Batch, Length, Model_Dim]`
    ///   - to:   `[Batch, Heads, Length, Head_Dim]`
    fn split_heads(&self, x: Tensor<B, 3>) -> Tensor<B, 4> {
        let [batch_size, sequence_length, _model_dimension] = x.dims();
        x.reshape([
            batch_size,
            sequence_length,
            self.head_count,
            self.head_dimension(),
        ])
        .swap_dims(1, 2)
    }

    /// - Reshapes and transposes `x`:
    ///   - from: `[Batch, Heads, Length, Head_Dim]`
    ///   - to:   `[Batch, Length, Model_Dim]`
    fn merge_heads(&self, x: Tensor<B, 4>) -> Tensor<B, 3> {
        let [batch_size, _head_count, sequence_length, _model_dimension] = x.dims();
        x.swap_dims(1, 2)
            .reshape([batch_size, sequence_length, self.model_dimension])
    }

    pub fn forward(
        &self,
        query: Tensor<B, 3>,
        key: Tensor<B, 3>,
        value: Tensor<B, 3>,
        mask: Option<Tensor<B, 4, Bool>>,
    ) -> Tensor<B, 3> {
        // initial weight projections
        let query = self.query.forward(query);
        let key = self.key.forward(key);
        let value = self.value.forward(value);

        let query = self.split_heads(query);
        let key = self.split_heads(key);
        let value = self.split_heads(value);

        // scaled dot product attention
        let context = self.scaled_dot_product_attention(query, key, value, mask);

        // final output layer
        let merged = self.merge_heads(context);
        let output = self.output_projection.forward(merged);

        output
    }
}
