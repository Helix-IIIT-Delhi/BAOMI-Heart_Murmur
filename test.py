# ===============================
# BanditAttention Layer Definition
# ===============================
class BanditAttention(tf.keras.layers.Layer):
    def __init__(self, num_heads, reward_decay=0.9, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.reward_decay = reward_decay
        # Initialize Q-values for each head
        self.q_values = self.add_weight(
            shape=(num_heads,),
            initializer='ones',
            trainable=False,
            name='q_values'
        )

    def call(self, attention_outputs, rewards):
        """
        attention_outputs: Tensor of shape (batch_size, seq_len, num_heads, head_dim)
        rewards: Tensor of shape (num_heads,)
        """
        # Normalize rewards
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        normalized_rewards = rewards / (tf.reduce_sum(rewards) + 1e-8)

        # Update Q-values (bandit reward mechanism)
        new_q_values = self.q_values * self.reward_decay + normalized_rewards * (1 - self.reward_decay)
        self.q_values.assign(new_q_values)

        # Select the best head based on Q-values
        best_head_idx = tf.argmax(self.q_values)

        # Gather the attention output from the selected head
        # attention_outputs shape: (batch_size, seq_len, num_heads, head_dim)
        selected_attention = attention_outputs[:, :, best_head_idx, :]

        return selected_attention  # Shape: (batch_size, seq_len, head_dim)

# ================================================
# Bandit Cross-Attention Layer Definition
# ================================================
class BanditCrossAttentionLayer(tf.keras.layers.Layer):
    def __init__(self, num_heads, head_dim=32, reward_decay=0.9, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.reward_decay = reward_decay
        self.feature_dim = num_heads * head_dim

        # Define Dense layers for projecting query, key, value
        self.query_dense = Dense(self.feature_dim, name='query_projection')
        self.key_dense = Dense(self.feature_dim, name='key_projection')
        self.value_dense = Dense(self.feature_dim, name='value_projection')

        # Define BanditAttention layer
        self.bandit_attention = BanditAttention(num_heads, reward_decay, name='bandit_attention')

    def build(self, input_shape):
        # Ensure that the input shapes are as expected
        # input_shape is a list: [query_shape, key_shape, value_shape]
        assert isinstance(input_shape, list), "Input should be a list of [query, key, value] tensors."
        query_shape, key_shape, value_shape = input_shape
        # Further assertions can be added here if necessary
        super().build(input_shape)

    def call(self, inputs):
        query, key, value = inputs  # All have shape (batch_size, seq_len, feature_dim)

        # Project inputs
        query_proj = self.query_dense(query)  # (batch_size, seq_len_q, feature_dim)
        key_proj = self.key_dense(key)        # (batch_size, seq_len_k, feature_dim)
        value_proj = self.value_dense(value)  # (batch_size, seq_len_v, feature_dim)

        # Reshape for multi-head attention
        # (batch_size, num_heads, seq_len_q, head_dim)
        query_reshaped = tf.reshape(query_proj, (-1, tf.shape(query_proj)[1], self.num_heads, self.head_dim))
        key_reshaped = tf.reshape(key_proj, (-1, tf.shape(key_proj)[1], self.num_heads, self.head_dim))
        value_reshaped = tf.reshape(value_proj, (-1, tf.shape(value_proj)[1], self.num_heads, self.head_dim))

        # Transpose to (batch_size, num_heads, seq_len, head_dim)
        query_reshaped = tf.transpose(query_reshaped, perm=[0, 2, 1, 3])
        key_reshaped = tf.transpose(key_reshaped, perm=[0, 2, 1, 3])
        value_reshaped = tf.transpose(value_reshaped, perm=[0, 2, 1, 3])

        # Compute attention scores (scaled dot-product attention)
        # Corrected einsum equation: 'bnqh,bnkh->bnqk'
        attention_scores = tf.einsum('bnqh,bnkh->bnqk', query_reshaped, key_reshaped) / tf.math.sqrt(tf.cast(self.head_dim, tf.float32))
        attention_weights = tf.nn.softmax(attention_scores, axis=-1)  # (batch_size, num_heads, seq_len_q, seq_len_k)

        # Apply attention weights to value
        # Corrected einsum equation: 'bnqk,bnkh->bnqh'
        attention_outputs = tf.einsum('bnqk,bnkh->bnqh', attention_weights, value_reshaped)  # (batch_size, num_heads, seq_len_q, head_dim)

        # Transpose back to (batch_size, seq_len_q, num_heads, head_dim)
        attention_outputs = tf.transpose(attention_outputs, perm=[0, 2, 1, 3])

        # Example rewards: could be based on some performance metric or external signal
        # Here, we use random rewards for demonstration; replace with actual rewards as needed
        rewards = tf.random.uniform((self.num_heads,), minval=0.0, maxval=1.0)

        # Apply BanditAttention layer
        attended = self.bandit_attention(attention_outputs, rewards)  # (batch_size, seq_len_q, head_dim)

        return attended  # Fused representation

    def compute_output_shape(self, input_shape):
        # input_shape is a list: [query_shape, key_shape, value_shape]
        query_shape, key_shape, value_shape = input_shape
        batch_size = query_shape[0]
        seq_len_q = query_shape[1]
        return (batch_size, seq_len_q, self.head_dim)
def build_model(input_shape1, input_shape2, num_heads=4, head_dim=32):
    # Input layers
    input1 = Input(shape=input_shape1, name='Input1')
    input2 = Input(shape=input_shape2, name='Input2')

    # -----------------------
    # First Branch
    # -----------------------
    branch1 = Conv1D(32, kernel_size=3, activation='relu')(input1)
    branch1 = BatchNormalization()(branch1)
    branch1 = MaxPooling1D(pool_size=2)(branch1)
    branch1 = Dropout(0.2)(branch1)
    branch1 = Conv1D(64, kernel_size=3, activation='relu')(branch1)
    branch1 = BatchNormalization()(branch1)
    branch1 = MaxPooling1D(pool_size=2)(branch1)
    branch1 = Dropout(0.2)(branch1)
    # At this point, branch1 has shape (batch_size, seq_len, 128)

    # -----------------------
    # Second Branch
    # -----------------------
    branch2 = Conv1D(32, kernel_size=3, activation='relu')(input2)
    branch2 = BatchNormalization()(branch2)
    branch2 = MaxPooling1D(pool_size=2)(branch2)
    branch2 = Dropout(0.2)(branch2)
    branch2 = Conv1D(64, kernel_size=3, activation='relu')(branch2)
    branch2 = BatchNormalization()(branch2)
    branch2 = MaxPooling1D(pool_size=2)(branch2)
    branch2 = Dropout(0.2)(branch2)
    # At this point, branch2 has shape (batch_size, seq_len, 128)

    # -----------------------
    # Bandit-Based Cross-Attention Fusion
    # -----------------------
    # Instantiate two separate attention layers for bidirectional attention
    attention_layer1 = BanditCrossAttentionLayer(num_heads=num_heads, head_dim=head_dim, name='bandit_cross_attention_1')
    attention_layer2 = BanditCrossAttentionLayer(num_heads=num_heads, head_dim=head_dim, name='bandit_cross_attention_2')

    # Apply cross-attention: branch1 attends to branch2 and vice versa
    attention1 = attention_layer1([branch1, branch2, branch2])  # (batch_size, seq_len_q, head_dim)
    attention2 = attention_layer2([branch2, branch1, branch1])  # (batch_size, seq_len_q, head_dim)

    # Pass the tensor outputs to Dense layers
    dense_layer1 = Dense(128, activation='relu')(attention1)  # (batch_size, seq_len_q, 128)
    dense_layer2 = Dense(128, activation='relu')(attention2)  # (batch_size, seq_len_q, 128)

    # Pooling instead of Flatten
    attention1_pooled = GlobalAveragePooling1D()(dense_layer1)  # (batch_size, 128)
    attention2_pooled = GlobalAveragePooling1D()(dense_layer2)  # (batch_size, 128)

    # Concatenate the pooled features
    fused = concatenate([attention1_pooled, attention2_pooled], axis=-1)  # Shape: (batch_size, 256)

    # -----------------------
    # Dense Layers for Classification
    # -----------------------
    # dense_layer = Dense(128, activation='relu')(fused)
    # dense_layer = Dropout(0.3)(dense_layer)
    dense_layer = Dense(64, activation='relu')(fused)
    dense_layer = Dropout(0.3)(dense_layer)
    output = Dense(2, activation='softmax', name="Output")(dense_layer)

    # Define the model
    model = Model(inputs=[input1, input2], outputs=output, name="Bandit_Cross_Attention_Conv1D_Model")

    return model

