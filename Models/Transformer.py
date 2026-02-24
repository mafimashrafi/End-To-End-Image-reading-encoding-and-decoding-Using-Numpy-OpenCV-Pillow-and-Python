import numpy as np 

class Transformer:
    def __init__(self, cnn_output_dim=16384, dimension_of_model=512, num_attention_head=8, num_e2d_layers=6):
        self.cnn_output_dim = cnn_output_dim
        self.dimension_of_model = dimension_of_model 
        self.num_attention_head = num_attention_head
        self.num_e2d_layers = num_e2d_layers
        self.d_head = dimension_of_model // num_attention_head
        
        self.initialize_projections()
        self.initialize_layers()
    
    def initialize_projections(self):
        """Initialize projection from CNN vector to dimension_of_model dimension"""
        # Project CNN feature vector to dimension_of_model
        self.W_cnn_proj = np.random.randn(self.cnn_output_dim, self.dimension_of_model) * 0.02
        self.b_cnn_proj = np.zeros((1, self.dimension_of_model))
        

        self.positional_embed = np.random.randn(512, self.dimension_of_model) * 0.02
    
    def initialize_layers(self):
        self.encoder_layers = []
        self.decoder_layers = []
        
        for _ in range(self.num_e2d_layers):
            encoder = {
                'W_q': np.random.randn(self.dimension_of_model, self.dimension_of_model) * 0.02,
                'W_k': np.random.randn(self.dimension_of_model, self.dimension_of_model) * 0.02,
                'W_v': np.random.randn(self.dimension_of_model, self.dimension_of_model) * 0.02,
                'W_o': np.random.randn(self.dimension_of_model, self.dimension_of_model) * 0.02,
                'W_fc1': np.random.randn(self.dimension_of_model, 2048) * 0.02,
                'W_fc2': np.random.randn(2048, self.dimension_of_model) * 0.02,
                'b_fc1': np.zeros((1, 2048)),
                'b_fc2': np.zeros((1, self.dimension_of_model))
            }
            
            decoder = {**encoder}
            decoder['W_cq'] = np.random.randn(self.dimension_of_model, self.dimension_of_model) * 0.02
            decoder['W_ck'] = np.random.randn(self.dimension_of_model, self.dimension_of_model) * 0.02
            decoder['W_cv'] = np.random.randn(self.dimension_of_model, self.dimension_of_model) * 0.02
            decoder['W_co'] = np.random.randn(self.dimension_of_model, self.dimension_of_model) * 0.02
            
            self.encoder_layers.append(encoder)
            self.decoder_layers.append(decoder)
    
    @staticmethod
    def relu(x):
        return np.maximum(0, x)
    
    @staticmethod
    def relu_derivative(x):
        return (x > 0).astype(float)
    
    @staticmethod
    def softmax(x):
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    @staticmethod
    def layer_normalization(x, eps=1e-6):
        mean = np.mean(x, axis=-1, keepdims=True)
        std = np.std(x, axis=-1, keepdims=True)
        return (x - mean) / (std + eps)
    
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """
        Scaled dot-product attention mechanism
        Q: Query matrix (batch_size, seq_len, dimension_of_model)
        K: Key matrix (batch_size, seq_len, dimension_of_model)
        V: Value matrix (batch_size, seq_len, dimension_of_model)
        """
        # Compute attention scores
        scores = np.matmul(Q, K.transpose(0, 2, 1)) / np.sqrt(self.d_head)
        
        if mask is not None:
            scores = scores + mask * -1e9
        
        attention_weights = self.softmax(scores)
        
        output = np.matmul(attention_weights, V)
        
        return output, attention_weights
    
    def multi_head_attention(self, X, W_q, W_k, W_v, W_o):
        """
        Multi-head attention
        X: Input (batch_size, seq_len, dimension_of_model)
        """
        batch_size, seq_len, _ = X.shape
        
        # Linear projections
        Q = np.matmul(X, W_q)  # (batch_size, seq_len, dimension_of_model)
        K = np.matmul(X, W_k)
        V = np.matmul(X, W_v)
        
        # Reshape for multi-head
        Q = Q.reshape(batch_size, seq_len, self.num_attention_head, self.d_head)
        K = K.reshape(batch_size, seq_len, self.num_attention_head, self.d_head)
        V = V.reshape(batch_size, seq_len, self.num_attention_head, self.d_head)
        
        Q = Q.transpose(0, 2, 1, 3)
        K = K.transpose(0, 2, 1, 3)
        V = V.transpose(0, 2, 1, 3)
        
        attn_output = np.zeros_like(Q)
        for h in range(self.num_attention_head):
            attn, _ = self.scaled_dot_product_attention(Q[:, h], K[:, h], V[:, h])
            attn_output[:, h] = attn
        
        attn_output = attn_output.transpose(0, 2, 1, 3)
        attn_output = attn_output.reshape(batch_size, seq_len, self.dimension_of_model)

        output = np.matmul(attn_output, W_o)
        
        return output
    
    def feed_forward(self, X, W_fc1, W_fc2, b_fc1, b_fc2):
        """
        Feed-forward network (Position-wise Feed-Forward Networks)
        """
        hidden = np.matmul(X, W_fc1) + b_fc1
        hidden = self.relu(hidden)
        output = np.matmul(hidden, W_fc2) + b_fc2
        return output
    
    def encoder_block(self, X, layer_params):
        """
        Single encoder block with self-attention + feed-forward
        """
        # Self-attention
        attn_output = self.multi_head_attention(
            X, 
            layer_params['W_q'], 
            layer_params['W_k'], 
            layer_params['W_v'], 
            layer_params['W_o']
        )
        
        X = self.layer_normalization(X + attn_output)
        
        feed_forward_output = self.feed_forward(
            X,
            layer_params['W_fc1'],
            layer_params['W_fc2'],
            layer_params['b_fc1'],
            layer_params['b_fc2']
        )
        
        X = self.layer_normalization(X + feed_forward_output)
        
        return X
    
    def decoder_block(self, X, encoder_output, layer_params):
        """
        Single decoder block with masked self-attention + cross-attention + feed-forward
        """
        attn_output = self.multi_head_attention(
            X, 
            layer_params['W_q'], 
            layer_params['W_k'], 
            layer_params['W_v'], 
            layer_params['W_o']
        )
        X = self.layer_normalization(X + attn_output)
        
        # Cross-attention with encoder output
        cross_attn = self.multi_head_attention(
            X,
            layer_params['W_cq'],
            layer_params['W_ck'],
            layer_params['W_cv'],
            layer_params['W_co']
        )
        X = self.layer_normalization(X + cross_attn)
        
        # Feed-forward
        feed_forward_output = self.feed_forward(
            X,
            layer_params['W_fc1'],
            layer_params['W_fc2'],
            layer_params['b_fc1'],
            layer_params['b_fc2']
        )
        X = self.layer_normalization(X + feed_forward_output)
        
        return X
    
    def encode(self, cnn_vector):
        """
        Encoder forward pass - projects CNN vector and adds positional encoding
        cnn_vector: CNN output vector (batch_size, cnn_output_dim)
        Returns: encoded representation (batch_size, dimension_of_model)
        """
        X = np.matmul(cnn_vector, self.W_cnn_proj) + self.b_cnn_proj
        X = X + self.positional_embed[0:1]
        
        for layer_params in self.encoder_layers:
            X = self.encoder_block(X, layer_params)
        
        return X
    
    def decode(self, output_tokens, encoder_output):
        """
        Decoder forward pass - projects output tokens and attends to encoder
        output_tokens: (batch_size, seq_len) token indices
        encoder_output: encoder output from encode() (batch_size, dimension_of_model)
        Returns: decoded representation (batch_size, seq_len, dimension_of_model)
        """
        batch_size, seq_len = output_tokens.shape
        
        if not hasattr(self, 'W_output_embed'):
            self.W_output_embed = np.random.randn(1000, self.dimension_of_model) * 0.02
        
        X = self.W_output_embed[output_tokens] + self.positional_embed[:seq_len]

        encoder_output_expanded = np.repeat(encoder_output, seq_len, axis=0).reshape(batch_size, seq_len, self.dimension_of_model)

        for layer_params in self.decoder_layers:
            X = self.decoder_block(X, encoder_output_expanded, layer_params)
        
        return X
    
    def forward(self, cnn_vector, output_tokens):
        """
        Forward pass through transformer
        cnn_vector: CNN output vector (batch_size, cnn_output_dim)
        output_tokens: (batch_size, seq_len) output token indices
        Returns: decoded output (batch_size, seq_len, dimension_of_model)
        """
        encoder_output = self.encode(cnn_vector)

        decoder_output = self.decode(output_tokens, encoder_output)
        
        return decoder_output