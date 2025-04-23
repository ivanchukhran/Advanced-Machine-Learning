import os
import random
import re
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import Flickr8k

# Set random seeds for reproducibility
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Image preprocessing
transform = transforms.Compose(
    [transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]
)


class Vocabulary:
    def __init__(self, freq_threshold=5):
        self.itos = {0: "[PAD]", 1: "[START]", 2: "[END]", 3: "[UNK]"}
        self.stoi = {"[PAD]": 0, "[START]": 1, "[END]": 2, "[UNK]": 3}
        self.freq_threshold = freq_threshold

    def __len__(self):
        return len(self.itos)

    def build_vocabulary(self, sentence_list):
        frequencies = Counter()
        idx = 4

        for sentence in sentence_list:
            for word in sentence.split():
                frequencies[word] += 1

                if word not in self.stoi and frequencies[word] >= self.freq_threshold:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1

    def numericalize(self, text):
        tokenized_text = text.split()

        return [self.stoi[token] if token in self.stoi else self.stoi["[UNK]"] for token in tokenized_text]

    def decode(self, indices):
        return [self.itos[idx] for idx in indices]


class FlickrDatasetWrapper(Dataset):
    def __init__(self, root_dir, ann_file, transform=None, freq_threshold=5):
        """
        A wrapper around torchvision's Flickr8k dataset to make it compatible with our model

        Args:
            root_dir: Directory with Flickr8k images
            ann_file: Path to the annotations file
            transform: Image transformations
            freq_threshold: Minimum word frequency threshold for vocabulary
        """
        self.flickr_dataset = Flickr8k(
            root=root_dir,
            ann_file=ann_file,
            transform=transform,
        )

        # Extract all captions
        self.captions = []
        for _, captions in self.flickr_dataset.annotations:
            for caption in captions:
                # Convert caption to lowercase and remove special characters
                caption = caption.lower()
                caption = re.sub(r"[^\w\s]", "", caption)
                self.captions.append(caption)

        # Initialize and build vocabulary
        self.vocab = Vocabulary(freq_threshold)
        self.vocab.build_vocabulary(self.captions)

    def __len__(self):
        return len(self.flickr_dataset)

    def __getitem__(self, idx):
        image, captions = self.flickr_dataset[idx]

        # Use the first caption (could randomly choose one instead)
        caption = captions[0].lower()
        caption = re.sub(r"[^\w\s]", "", caption)

        # Add START and END tokens
        caption = "[START] " + caption + " [END]"

        # Numericalize the caption
        caption_vec = self.vocab.numericalize(caption)

        return image, torch.tensor(caption_vec)


# You can still use the original FlickrDataset if you prefer or need more control
class FlickrDataset(Dataset):
    def __init__(self, root_dir, captions_file, transform=None, freq_threshold=5):
        self.root_dir = root_dir
        self.transform = transform

        self.imgs = []
        self.captions = []
        with open(captions_file, "r") as f:
            # self.annotations = f.read()
            next(f)  # skip header
            for i, line in enumerate(f):
                print(f"line: {i}: {line}")
                parts = line.split(",")
                img = parts[0]
                caption = parts[1]
                if len(parts) != 2:
                    caption = "".join(parts[1:-1])
                if caption.startswith('"') and caption.endswith('"'):
                    caption = caption[1:-1]
                self.imgs.append(img)
                self.captions.append(caption)

        # Initialize and build vocabulary
        self.vocab = Vocabulary(freq_threshold)
        self.vocab.build_vocabulary(self.captions)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_name = self.imgs[idx]
        caption = self.captions[idx]

        img_path = os.path.join(self.root_dir, img_name)
        image = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        # Add START and END tokens
        caption = "[START] " + caption + " [END]"

        # Numericalize the caption
        caption_vec = self.vocab.numericalize(caption)

        return image, torch.tensor(caption_vec)


# Collate function for DataLoader
def collate_fn(batch):
    imgs = [item[0].unsqueeze(0) for item in batch]
    imgs = torch.cat(imgs, dim=0)

    targets = [item[1] for item in batch]
    targets = nn.utils.rnn.pad_sequence(targets, batch_first=True, padding_value=0)

    return imgs, targets


# CNN Feature Extractor
class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        # Use pretrained MobileNet v2
        mobilenet = torchvision.models.mobilenet_v2(weights="DEFAULT")
        self.mobilenet = mobilenet.features
        # modules = list(mobilenet.children())[:-1]
        # self.mobilenet = nn.Sequential(*modules)

        # Feature projection
        self.projection = nn.Linear(mobilenet.last_channel, embed_size)
        self.dropout = nn.Dropout(0.5)

    def forward(self, images):
        # Extract features
        with torch.no_grad():
            features = self.mobilenet(images)  # [batch_size, channels, height, width]

        # Get dimensions
        batch_size = features.size(0)
        channels = features.size(1)
        height = features.size(2)
        width = features.size(3)

        # Reshape to [batch_size, height*width, channels]
        features = features.permute(0, 2, 3, 1)  # [batch_size, height, width, channels]
        features = features.reshape(batch_size, height * width, channels)  # [batch_size, height*width, channels]

        # Project each position to embed_size
        features = self.dropout(self.projection(features))

        return features

    # def forward(self, images):
    #     # Extract features
    #     with torch.no_grad():
    #         features = self.mobilenet(images)

    #     # Reshape features
    #     features = features.reshape(features.size(0), -1)

    #     # Project features
    #     features = self.dropout(self.projection(features))

    #     # Reshape for attention: (batch_size, h*w, embed_size)
    #     features = features.unsqueeze(1)

    #     return features


# Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length=50):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:, : x.size(1), :]


# Sequence Embedding Layer
class SeqEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_size, max_seq_length=50):
        super(SeqEmbedding, self).__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.position_encoding = PositionalEncoding(embed_size, max_seq_length)

    def forward(self, x):
        x = self.token_embedding(x)
        x = self.position_encoding(x)
        return x


# Causal Self-Attention Layer
class CausalSelfAttention(nn.Module):
    def __init__(self, embed_size, num_heads, dropout=0.1):
        super(CausalSelfAttention, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_size, num_heads, dropout=dropout, batch_first=True)
        self.layer_norm = nn.LayerNorm(embed_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Create causal attention mask
        sz = x.size(1)
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float("-inf")).masked_fill(mask == 1, float(0.0))
        mask = mask.to(x.device)

        # Self-attention
        attn_output, _ = self.multihead_attn(x, x, x, attn_mask=mask)

        # Add & Norm
        x = x + self.dropout(attn_output)
        x = self.layer_norm(x)

        return x


# Cross-Attention Layer
class CrossAttention(nn.Module):
    def __init__(self, embed_size, num_heads, dropout=0.1):
        super(CrossAttention, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_size, num_heads, dropout=dropout, batch_first=True)
        self.layer_norm = nn.LayerNorm(embed_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key_value):
        # Cross-attention
        attn_output, attn_weights = self.multihead_attn(query, key_value, key_value)

        # Add & Norm
        out = query + self.dropout(attn_output)
        out = self.layer_norm(out)

        # Store attention weights for visualization
        self.last_attention_scores = attn_weights

        return out


# Feed Forward Layer
class FeedForward(nn.Module):
    def __init__(self, embed_size, ff_size, dropout=0.1):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(embed_size, ff_size)
        self.linear2 = nn.Linear(ff_size, embed_size)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embed_size)

    def forward(self, x):
        # FFN
        ff_output = self.dropout(self.linear2(F.relu(self.linear1(x))))

        # Add & Norm
        out = x + ff_output
        out = self.layer_norm(out)

        return out


# Decoder Layer
class DecoderLayer(nn.Module):
    def __init__(self, embed_size, num_heads, feedforward_size, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.self_attention = CausalSelfAttention(embed_size, num_heads, dropout)
        self.cross_attention = CrossAttention(embed_size, num_heads, dropout)
        self.feed_forward = FeedForward(embed_size, feedforward_size, dropout)

    def forward(self, text, image):
        # Self-attention
        text = self.self_attention(text)

        # Cross-attention
        text = self.cross_attention(text, image)

        # Feed Forward
        text = self.feed_forward(text)

        return text


# Decoder (Transformer)
class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_size, num_layers, num_heads, feedforward_size, max_seq_length=50, dropout=0.1):
        super(Decoder, self).__init__()

        # Embedding layer
        self.seq_embedding = SeqEmbedding(vocab_size, embed_size, max_seq_length)

        # Decoder layers
        self.decoder_layers = nn.ModuleList(
            [DecoderLayer(embed_size, num_heads, feedforward_size, dropout) for _ in range(num_layers)]
        )

        # Output layer
        self.output_layer = nn.Linear(embed_size, vocab_size)

    def forward(self, text, image):
        # Embedding
        text = self.seq_embedding(text)

        # Process through decoder layers
        for layer in self.decoder_layers:
            text = layer(text, image)

        # Convert to vocabulary size
        output = self.output_layer(text)

        return output


# Complete Captioning Model
class ImageCaptioningModel(nn.Module):
    def __init__(
        self, vocab_size, embed_size=256, num_layers=2, num_heads=8, feedforward_size=512, max_seq_length=50, dropout=0.1
    ):
        super(ImageCaptioningModel, self).__init__()

        # Image encoder
        self.encoder = EncoderCNN(embed_size)

        # Text decoder
        self.decoder = Decoder(
            vocab_size=vocab_size,
            embed_size=embed_size,
            num_layers=num_layers,
            num_heads=num_heads,
            feedforward_size=feedforward_size,
            max_seq_length=max_seq_length,
            dropout=dropout,
        )

    def forward(self, images, captions):
        # Extract image features
        image_features = self.encoder(images)

        # Generate captions
        outputs = self.decoder(captions, image_features)

        return outputs

    def generate_caption(self, image, vocabulary, max_length=50, temperature=1.0):
        # Set model to evaluation mode
        self.eval()

        with torch.no_grad():
            # Encode image
            image_features = self.encoder(image.unsqueeze(0).to(device))

            # Initialize caption with START token
            caption = torch.tensor([[1]]).to(device)  # START token

            # Store attention maps
            attention_maps = []

            # Generate caption iteratively
            for i in range(max_length):
                output = self.decoder(caption, image_features)

                # Get prediction for the next word
                preds = output[:, -1, :]

                if temperature == 0:
                    # Greedy search
                    predicted_id = torch.argmax(preds, dim=-1, keepdim=True)
                else:
                    # Sample with temperature
                    preds = preds / temperature
                    predicted_id = torch.multinomial(F.softmax(preds, dim=-1), 1)

                # Append the predicted word to the caption
                caption = torch.cat([caption, predicted_id], dim=-1)

                # Store cross-attention scores from the last decoder layer
                attention_maps.append(self.decoder.decoder_layers[-1].cross_attention.last_attention_scores)

                # End if END token is predicted
                if predicted_id.item() == 2:  # END token
                    break

            # Convert indices to tokens
            tokens = caption[0].cpu().numpy().tolist()
            caption_text = " ".join([vocabulary.itos[idx] for idx in tokens[1:-1]])  # Skip START and END

            return caption_text, attention_maps


# Function to train the model
def train_model(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    vocab_size: int,
    vocabulary,
):
    model.train()

    for epoch in range(epochs):
        total_loss = 0
        total_correct = 0
        total_tokens = 0

        for i, (images, captions) in enumerate(dataloader):
            images = images.to(device)
            captions = captions.to(device)

            # Input tokens are all but the last token
            inputs = captions[:, :-1]

            # Target tokens are all but the first token
            targets = captions[:, 1:]

            # Forward pass
            outputs = model(images, inputs)

            # Reshape outputs and targets for loss calculation
            outputs = outputs.reshape(-1, vocab_size)
            targets = targets.reshape(-1)

            # Create mask (ignore padding tokens)
            mask = (targets != 0).float()

            # Calculate loss with masking
            loss = criterion(outputs, targets)
            masked_loss = (loss * mask).sum() / mask.sum() if mask.sum() > 0 else 0

            # Backpropagation
            optimizer.zero_grad()
            masked_loss.backward()
            optimizer.step()

            # Calculate accuracy
            _, predicted = outputs.max(1)
            correct = (predicted == targets).float() * mask
            total_correct += correct.sum().item()
            total_tokens += mask.sum().item()

            total_loss += masked_loss.item()

            if (i + 1) % 100 == 0:
                print(f"Epoch [{epoch + 1}/{epochs}], Step [{i + 1}/{len(dataloader)}], Loss: {masked_loss.item():.4f}")

        # Calculate epoch statistics
        avg_loss = total_loss / len(dataloader)
        accuracy = total_correct / total_tokens if total_tokens > 0 else 0

        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")

        # Generate sample caption
        sample_image = next(iter(dataloader))[0][0].to(device)
        model.eval()
        with torch.no_grad():
            for temp in [0.0, 0.5, 1.0]:
                caption, _ = model.generate_caption(sample_image, vocabulary, temperature=temp)
                print(f"Sample caption (temp={temp}): {caption}")
                model.train()


# Function to visualize attention maps
def plot_attention_maps(image, caption, attention_maps):
    # Prepare the figure
    fig = plt.figure(figsize=(16, 9))
    words = caption.split() + ["[END]"]

    # Process attention maps
    n_cols = min(6, len(words))
    n_rows = (len(words) + n_cols - 1) // n_cols

    for i, (word, attn_map) in enumerate(zip(words, attention_maps)):
        ax = fig.add_subplot(n_rows, n_cols, i + 1)
        ax.set_title(word)

        # Display image
        img = image.squeeze(0).permute(1, 2, 0).cpu().numpy()
        img = (img - img.min()) / (img.max() - img.min())
        ax.imshow(img)

        # Process and display attention map
        attn = attn_map.squeeze(0).cpu().numpy()  # Remove batch dimension
        attn = np.mean(attn, axis=0)  # Average across heads
        attn = attn.reshape(7, 7)  # Reshape to 7x7 grid

        # Normalize attention weights
        attn = (attn - attn.min()) / (attn.max() - attn.min() + 1e-9)

        ax.imshow(attn, cmap="hot", alpha=0.3, extent=(0, img.shape[1], img.shape[0], 0))
        ax.axis("off")

    plt.tight_layout()
    plt.suptitle(caption, fontsize=16)
    plt.subplots_adjust(top=0.9)

    return fig


# Example usage (uncomment when ready to use)
"""
# Option 1: Using torchvision's Flickr8k dataset
root_dir = 'path/to/flickr8k'  # Main directory containing 'Images' folder
ann_file = 'path/to/flickr8k/captions.txt'  # Path to captions file

# Create dataset and dataloader using the wrapper
dataset = FlickrDatasetWrapper(root_dir, ann_file, transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)

# Option 2: Using custom JSON format
# root_dir = 'path/to/flickr8k/images'
# captions_file = 'path/to/flickr8k/captions.json'
# dataset = FlickrDataset(root_dir, captions_file, transform=transform)
# dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)

# Initialize model
vocab_size = len(dataset.vocab)
model = ImageCaptioningModel(vocab_size).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss(reduction='none')
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# Train model
train_model(model, dataloader, criterion, optimizer, epochs=30, vocab_size=vocab_size)

# Save model
torch.save(model.state_dict(), 'image_captioning_model.pth')

# Test with a sample image
sample_image = Image.open('path/to/test/image.jpg').convert('RGB')
sample_image = transform(sample_image).unsqueeze(0).to(device)

# Generate caption
model.eval()
caption, attention_maps = model.generate_caption(sample_image, dataset.vocab)
print(f"Generated caption: {caption}")

# Visualize attention
plot_attention_maps(sample_image, caption, attention_maps)
plt.savefig('attention_visualization.png')
plt.show()
"""


# Function to load and preprocess a single image
def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image)
    return image


# Function to demonstrate the model on a given image
def demonstrate_captioning(model, image_path, vocab):
    # Load and preprocess image
    image = load_image(image_path).to(device)

    # Generate captions with different temperatures
    results = []
    attention_maps = []

    for temp in [0.0, 0.5, 1.0]:
        caption, attn_maps = model.generate_caption(image, vocab, temperature=temp)
        results.append(f"Temperature {temp}: {caption}")

        if temp == 0.0:  # Save attention maps for greedy search
            attention_maps = attn_maps

    # Visualize attention for greedy search result
    caption = results[0].split(": ")[1]
    fig = plot_attention_maps(image, caption, attention_maps)

    return results, fig


def main():
    root_dir = "/home/ivan/datasets/Flickr8k"
    if not os.path.exists(root_dir):
        os.mkdir(root_dir)
    captions_file = os.path.join(root_dir, "captions.txt")
    dataset_file = os.path.join(root_dir, "Images")

    # Create dataset and dataloader
    dataset = FlickrDataset(dataset_file, captions_file, transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)

    # Initialize model
    vocab_size = len(dataset.vocab)
    model = ImageCaptioningModel(vocab_size).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(reduction="none")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    # Train model
    train_model(model, dataloader, criterion, optimizer, epochs=30, vocab_size=vocab_size, vocabulary=dataset.vocab)

    # Save model
    torch.save(model.state_dict(), "image_captioning_model.pth")

    model.load_state_dict(torch.load("image_captioning_model.pth"))

    # Test with a sample image
    sample_image = Image.open("/home/ivan/Downloads/27a051.jpeg").convert("RGB")
    image_tensor = transform(sample_image)
    sample_image = image_tensor.to(device)

    # Generate caption
    model.eval()
    caption, attention_maps = model.generate_caption(sample_image, dataset.vocab)
    print(f"Generated caption: {caption}")

    # Visualize attention
    plot_attention_maps(sample_image, caption, attention_maps)
    plt.savefig("attention_visualization.png")
    plt.show()


if __name__ == "__main__":
    main()
