from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

class Embedding():
    
    def __init__(self, docs=None, tokenizer=None, model=None):
        self.docs = docs
        if tokenizer is None or model is None:
            self.tokenizer, self.model = self._model_tokenizer()
        else:
            self.tokenizer = tokenizer
            self.model = model
    
    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    def _model_tokenizer(self, tokenizer, model):
        # Load model from HuggingFace Hub
        tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L12-v2')
        model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L12-v2')
        return tokenizer, model
    
    def forward(self, docs=None):
        if docs is not None:
            self.docs = docs
        
        # Tokenize sentences
        encoded_input = self.tokenizer(self.docs, padding=True,truncation=True,return_tensors='pt')    
        
        # Compute the token embedding
        with torch.no_grad():
            model_output = self.model(**encoded_input)
        
        # Perform pooling
        sentence_embeddings = self._mean_pooling(model_output, encoded_input['attention_mask'])
        
        # Normalize embeddings
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        
        return sentence_embeddings        