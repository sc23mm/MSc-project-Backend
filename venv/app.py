from flask import Flask, request, jsonify
import torch
import os
from transformers import RobertaTokenizer, RobertaModel, RobertaConfig
from torch import nn
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Define the emotion labels
id2label = {
    0: "admiration", 1: "amusement", 2: "anger", 3: "annoyance", 4: "approval", 5: "caring",
    6: "confusion", 7: "curiosity", 8: "desire", 9: "disappointment", 10: "disapproval",
    11: "disgust", 12: "embarrassment", 13: "excitement", 14: "fear", 15: "gratitude",
    16: "grief", 17: "joy", 18: "love", 19: "nervousness", 20: "optimism", 21: "pride",
    22: "realization", 23: "relief", 24: "remorse", 25: "sadness", 26: "surprise", 27: "neutral"
}

# Directory where the model and tokenizer are saved
model_dir = "/Users/mukundhanmohan/Downloads/saved_model_hybrid_one"

# Load tokenizer
tokenizer = RobertaTokenizer.from_pretrained(model_dir)

# Define the model class
class RoBERTaRNNCNNFNNTransformer(nn.Module):
    def __init__(self, rnn_hidden_dim, cnn_out_channels, cnn_kernel_size, transformer_layers, ffn_hidden_dim, output_dim):
        super(RoBERTaRNNCNNFNNTransformer, self).__init__()
        self.roberta = RobertaModel.from_pretrained(model_dir)
        self.rnn = nn.LSTM(input_size=768, hidden_size=rnn_hidden_dim, batch_first=True, bidirectional=True)
        self.conv1d = nn.Conv1d(in_channels=rnn_hidden_dim * 2, out_channels=cnn_out_channels, kernel_size=cnn_kernel_size)
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=cnn_out_channels, nhead=8)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_layer, num_layers=transformer_layers)
        self.ffn = nn.Sequential(
            nn.Linear(cnn_out_channels, ffn_hidden_dim),
            nn.ReLU(),
            nn.Linear(ffn_hidden_dim, output_dim)
        )

    def forward(self, input_ids, attention_mask):
        roberta_output = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = roberta_output.last_hidden_state
        rnn_output, _ = self.rnn(sequence_output)
        rnn_output = rnn_output.permute(0, 2, 1)
        conv_output = self.conv1d(rnn_output)
        pooled_output = torch.max(conv_output, dim=2)[0]
        transformer_output = self.transformer_encoder(pooled_output.unsqueeze(1))
        pooled_transformer_output = transformer_output.mean(dim=1)
        logits = self.ffn(pooled_transformer_output)
        return logits

# Initialize model with the same architecture used during training
model = RoBERTaRNNCNNFNNTransformer(
    rnn_hidden_dim=128,
    cnn_out_channels=64,
    cnn_kernel_size=3,
    transformer_layers=2,
    ffn_hidden_dim=32,
    output_dim=len(id2label)
)

# Load the trained model's state dict
model.load_state_dict(torch.load(os.path.join(model_dir, 'pytorch_model.bin'), map_location=torch.device('cpu')))

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data['text']
    
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        _, predicted_class = torch.max(outputs, dim=1)
    
    predicted_label = id2label[predicted_class.item()]
    return jsonify({"label": predicted_label})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
