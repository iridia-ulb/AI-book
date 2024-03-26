import torch
import torch.nn as nn
from torch.nn import functional as F
import argparse

# Initialisation de la seed pour la reproductibilité
torch.manual_seed(1337)

# GPU ou CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device: ", device)

# Démarrons avec le plus simple réseau de neurones qui soit.

## __init__ c'est le constructeur qui initialise les paramètres du réseau
## Dans PyTorch, cette méthode est utilisée pour définir l'architecture du réseau
## les nombres de couches, les filtres convolutifs,.... les couches linéaires,
## fonctions d'activation.
## La méthode "forward" définit ce qui se passe lorsqu'un input est présenté au réseau:
## l'étape d'inférence du réseau de neurones.


# Structure du modèle
class BigramLanguageModel(nn.Module):
    def __init__(self, n_embd, block_size, n_head, n_layer, dropout, vocab_size):
        super().__init__()
        self.block_size = block_size
        # chaque caractère devient un vecteur de dimension "n_embd"
        # On code les caractères
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        # On code aussi la position des caractères
        self.position_embedding_table = nn.Embedding(block_size, n_embd)

        self.blocks = nn.Sequential(
            *[Block(n_embd, n_head, dropout, block_size) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape  # B=Batch_Size, T=Block_Size

        # On va commencer à opérer ici avec
        # une matrice (batch_size, block_size, vocab_size)

        # Ici génère juste un prochain caractère
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        # On somme le caractère et sa position
        x = tok_emb + pos_emb
        # On passe le tout à travers les têtes d'attention empilées
        x = self.blocks(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            # l'input devient une matrice (batch_size*block_size, n_embd)
            # l'output juste un vecteur (batch_size*block_size) décalé d'une position par rapport à l'input
            targets = targets.view(B * T)
            # la loss fonction est calculée comme la cross_entropy
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            # l'input du modèle est toujours idx
            idx_cond = idx[:, -self.block_size :]
            # On calcule la prédiction du prochain caractère
            logits, loss = self(idx_cond)
            # On prend le dernier caractère du tableau car c'est celui qu'on essaie de prédire
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            # On concatène ce caractère à l'input et on boucle....
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


# La constitution d'un bloc, on prend la sortie des têtes et on les additionne directement avec l'input
class Block(nn.Module):
    def __init__(self, n_embd, n_head, dropout, block_size):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size, n_embd, block_size, dropout)
        self.ffwd = FeedForward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


# La concaténation de plusieurs têtes d'attention sur une même couche.
# Chacune travaille dans une partie de l'espace d'embedding différent
class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, head_size, n_embd, block_size, dropout):
        super().__init__()
        self.heads = nn.ModuleList(
            [Head(head_size, n_embd, block_size, dropout) for _ in range(n_head)]
        )
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # On concatène la sortie des têtes
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        return out


# Création d'une tête d'attention
class Head(nn.Module):
    def __init__(self, head_size, n_embd, block_size, dropout):
        super().__init__()
        self.head_size = head_size
        # Les deux matrices Key et Query de dimension (n_embd, head_size)
        # Elles projettent les caractères dans un espace de dimension head_size
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(
            dropout
        )  # Le dropout dont nous discutons dans le livre

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * self.head_size**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out


# La projection de la sortie de cette couche de tête d'attentions
class FeedForward(nn.Module):
    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


# Fonction pour obtenir un batch de données pour l'entraînement
def get_batch(split, block_size, batch_size, train_data, val_data):
    # Obtention d'un seul batch à partir d'un ofset aléatoire
    # et des block_size caractères séparés en x et en y
    # Un batch devient une matrice (batch_size x block_size)
    data = train_data if split == "train" else val_data
    # ix est un vecteur de 4 ofset.
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


# Calcul de la perte moyenne sur un seul batch
@torch.no_grad()
def estimate_loss(model, eval_iters, block_size, batch_size, train_data, val_data):
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(split, block_size, batch_size, train_data, val_data)
            logits, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


# Entraîne le modèle
def train(
    model,
    optimizer,
    max_iters,
    eval_iters,
    eval_interval,
    block_size,
    batch_size,
    train_data,
    val_data,
):
    for iter in range(max_iters):
        if iter % eval_interval == 0:
            losses = estimate_loss(
                model, eval_iters, block_size, batch_size, train_data, val_data
            )
            print(
                f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
            )
        xb, yb = get_batch("train", block_size, batch_size, train_data, val_data)
        logits, loss = model(xb, yb)

        optimizer.zero_grad()  # Calcul du gradient
        loss.backward()  # Rétropropagation
        optimizer.step()  # Optimisation


# Effectue l'inférence avec le modèle
def inference(model, context, max_new_tokens):
    # Générer du texte en utilisant le modèle
    generated_text = model.generate(context, max_new_tokens)
    # Retourner le texte généré
    return generated_text


# Sauvegarde le modèle
def save_model(model, filepath):
    torch.save(model.state_dict(), filepath)
    print(f"Modèle enregistré sous {filepath}")


# Charge le modèle
def load_model(model, filepath):
    model.load_state_dict(torch.load(filepath))
    print(f"Modèle chargé depuis {filepath}")
    return model


def prepare_data(filename):
    with open(filename, "r", encoding="utf-8") as f:
        text = f.read()

    # Obtention d'une liste triée de tous les caractères contenus dans le texte
    chars = sorted(list(set(text)))
    vocab_size = len(chars)

    # On crée un dictionnaire qui associe un entier à chaque caractère et vice-versa
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    # Fonctions pour encoder et décoder les caractères
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: "".join([itos[i] for i in l])

    # Encodage du texte en entiers
    data = torch.tensor(encode(text), dtype=torch.long)

    # Séparation classique pour l'apprentissage des NN du dataset en deux parties
    # le training et le validation set (90% et 10%)
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]

    return train_data, val_data, vocab_size, decode


# Récupère les arguments précisés au lancelent du script
def parse_args():
    parser = argparse.ArgumentParser(
        description="Train and/or infer with a language model"
    )

    # Mode (train ou infer)
    parser.add_argument("--train", action="store_true", help="Mode entraînement")
    parser.add_argument("--infer", action="store_true", help="Mode inférence")

    # Sauvegarde et chargement du modèle
    parser.add_argument(
        "--save_model",
        type=str,
        default="mymodel.pth",
        help="Sauvegarde le modèle dans le fichier spécifié",
    )
    parser.add_argument(
        "--load_model", type=str, help="Charge le modèle depuis le fichier spécifié"
    )

    # fichier text de données d'entrainement
    parser.add_argument(
        "--input",
        type=str,
        help="Utilise les données d'entrainement depuis le fichier spécifié",
        default="input.txt",
    )

    # Paramètres du modèle
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Nombre d'I/O que le modèle doit apprendre par batch",
    )
    parser.add_argument(
        "--block_size",
        type=int,
        default=32,
        help="Longueur des séquences que le transformer doit apprendre",
    )
    parser.add_argument(
        "--max_iters",
        type=int,
        default=1000,
        help="Nombre d'itérations d'apprentissage",
    )
    parser.add_argument(
        "--eval_interval",
        type=int,
        default=300,
        help="Intervalle d'évaluation pendant l'entraînement",
    )
    parser.add_argument(
        "--learning_rate", type=float, default=1e-3, help="Taux d'apprentissage"
    )
    parser.add_argument(
        "--eval_iters", type=int, default=200, help="Nombre d'itérations d'évaluation"
    )
    parser.add_argument(
        "--n_embd",
        type=int,
        default=32,
        help="Dimension de l'espace dans lequel on projette les caractères",
    )
    parser.add_argument(
        "--n_head", type=int, default=8, help="Nombre de têtes d'attention"
    )
    parser.add_argument("--n_layer", type=int, default=6, help="Nombre de couches")
    parser.add_argument(
        "--dropout", type=float, default=0.2, help="Probabilité de dropout"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Paramètres du modèle
    batch_size = args.batch_size
    block_size = args.block_size
    max_iters = args.max_iters
    eval_interval = args.eval_interval
    learning_rate = args.learning_rate
    eval_iters = args.eval_iters
    n_embd = args.n_embd
    n_head = args.n_head
    n_layer = args.n_layer
    dropout = args.dropout
    filename = args.input

    train_data, val_data, vocab_size, decode = prepare_data(filename)

    # Mode entraînement
    if args.train:
        print("Mode entraînement")

        # Lancer l'entraînement du modèle
        model = BigramLanguageModel(
            n_embd, block_size, n_head, n_layer, dropout, vocab_size
        )
        m = model.to(device)

        # Choix de l'algorithme d'optimisation
        optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)
        train(
            m,
            optimizer,
            max_iters,
            eval_iters,
            eval_interval,
            block_size,
            batch_size,
            train_data,
            val_data,
        )

        # Enregistre le modèle si c'est demandé
        if args.save_model:
            save_model(model, args.save_model)

    # Mode inférence
    elif args.infer:
        print("Mode inférence")
        if args.load_model:
            # Charge le modèle
            model = load_model(
                BigramLanguageModel(
                    n_embd, block_size, n_head, n_layer, dropout, vocab_size
                ),
                args.load_model,
            )
            m = model.to(device)

            # Génération de texte à partir du réseau de neurones
            context = torch.zeros((1, 1), dtype=torch.long, device=device)
            print("context= ", context)
            generated_text = inference(model, context, max_new_tokens=500)
            print("Texte généré :")
            print(generated_text)
            print("Décodé : ")
            print(decode(generated_text[0].tolist()))

        else:
            print(
                "Aucun modèle spécifié pour l'inférence. Veuillez spécifier --load_model."
            )

    else:
        print("Aucun mode sélectionné. Veuillez spécifier --train ou --infer.")


if __name__ == "__main__":
    main()
