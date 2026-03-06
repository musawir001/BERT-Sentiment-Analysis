import torch
import gradio as gr
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load model from current folder
model_path = "."

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

labels = ["Negative 😠", "Neutral 😐", "Positive 😊"]

def predict_sentiment(text):

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True
    )

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.nn.functional.softmax(outputs.logits, dim=1)

    confidence, predicted_class = torch.max(probs, dim=1)

    sentiment = labels[predicted_class.item()]
    confidence = round(confidence.item() * 100, 2)

    return f"{sentiment} (Confidence: {confidence}%)"

# Gradio Interface
interface = gr.Interface(
    fn=predict_sentiment,
    inputs=gr.Textbox(lines=2, placeholder="Enter a sentence..."),
    outputs="text",
    title="BERT Sentiment Analysis",
    description="AI model that predicts sentiment using a fine-tuned BERT model."
)

interface.launch()
