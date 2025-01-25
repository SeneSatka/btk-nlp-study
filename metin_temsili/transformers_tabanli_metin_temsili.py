# import libraries
from transformers import AutoTokenizer,AutoModel
import torch
# modl ve tokenizer yükle
model_name="bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# input text (metni) tanımla
text = "Transforöers can be used for natural language processing."


# metni tokenlara çevirmek
inputs = tokenizer(text,return_tensors="pt") # pt = pytorh , çıktı pytorch tensoru olarak return edilir

# modeli, kullanarak metin temsili oluştur
with torch.no_grad(): # gradyanların hesaplanması durdurulur, böylece belleği daha verimli kullanırız.
    outputs = model(**inputs)

# modelin çıkışundan sonra gizli durumu alalım
last_hidden_state = outputs.last_hidden_state # tüm token çıktılarını almak için

# ilk tokenin embedding inin alalım ve print ettirelim
first_token_embedding = last_hidden_state[0,0,:].numpy()

print(f"Metin temsili: {first_token_embedding}")