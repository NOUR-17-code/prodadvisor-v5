# 🛍️ PRODADVISOR V5

**Application de forecasting et conseil produit pour boutique de mode**

PRODADVISOR est un système décisionnel hybride qui combine **prévision de la demande** (TimeGPT / Nixtla) et **recommandation de stock prescriptive générée par IA** (Llama 3.2 3B fine-tuné en QLoRA), exposés via un dashboard interactif **Streamlit**.

---

## ✨ Fonctionnalités

- **Dashboard interactif** : KPIs en temps réel (demande prévue, élasticité promo, taux de retour, causes de retour) par catégorie / marque / saison.
- **Simulateur "What-If"** : ajustement du taux de promotion via un curseur pour recalculer instantanément l'impact sur la demande prévue (modèle d'élasticité prix).
- **Prévision de séries temporelles** avec **TimeGPT** (variables exogènes : taux de démarque, prix courant), comparée à une baseline LSTM historique.
- **Recommandation prescriptive en français**, générée par un LLM (Llama 3.2 3B Instruct, fine-tuné via QLoRA/LoRA) à partir du contexte produit + de la prévision de demande + de l'historique des retours.
- **Onglet "Sous le capot" (Explainable AI)** : visualisation du prompt exact injecté dans le LLM, métriques d'évaluation TimeGPT vs LSTM (MAE, RMSE), état du pipeline MLOps.
- **Mode dégradé robuste** : si aucune clé API (Nixtla / Hugging Face) n'est configurée, l'application bascule automatiquement sur des données et réponses simulées réalistes, afin que la démo fonctionne toujours de bout en bout.

---

## 🏗️ Architecture du pipeline

```
1. prepare_data.py        →  Agrège fashion_boutique_cleaned.csv en séries temporelles
                              (par catégorie/semaine ou mois) + split train/test chronologique
                              → train_time_series.csv / test_time_series.csv

2. forecast_timegpt.py    →  Appelle l'API TimeGPT (Nixtla) avec variables exogènes
                              (markdown_percentage, current_price), évalue vs baseline LSTM
                              → timegpt_predictions.csv

3. train_lora.py          →  Fine-tuning QLoRA de Llama 3.2 3B Instruct sur le dataset
                              prompt/réponse (train.jsonl / val.jsonl / test.jsonl)
                              → adaptateurs LoRA (prodadvisor_llama_adapters/)

4. app.py                 →  Interface Streamlit : dashboard, simulateur What-If,
                              recommandation LLM (API Hugging Face ou simulation), XAI
```

---

## 📁 Structure du dépôt

| Fichier / Dossier | Rôle |
|---|---|
| `app.py` | Application Streamlit principale (dashboard + LLM + XAI) |
| `prepare_data.py` | Préparation et agrégation des séries temporelles |
| `forecast_timegpt.py` | Prévision de la demande via TimeGPT (Nixtla) |
| `train_lora.py` | Fine-tuning QLoRA de Llama 3.2 3B (à exécuter sur GPU, ex. Kaggle) |
| `fashion_boutique_cleaned.csv` | Dataset principal nettoyé (ventes, retours, prix, etc.) |
| `train_time_series.csv` / `test_time_series.csv` | Séries temporelles agrégées (généré par `prepare_data.py`) |
| `timegpt_predictions.csv` | Prévisions TimeGPT pré-calculées (utilisées par l'app) |
| `train.jsonl` / `val.jsonl` / `test.jsonl` | Dataset instruction/réponse pour le fine-tuning LLM |
| `requirements.txt` | Dépendances Python pour l'application Streamlit |
| `.gitignore` | Exclut `.streamlit/secrets.toml` et `__pycache__/` |
| `.streamlit/secrets.toml` *(non versionné)* | Clés API privées (HF_TOKEN, NIXTLA_API_KEY) |

---

## ⚙️ Installation locale

```bash
# 1. Cloner le dépôt
git clone https://github.com/NOUR-17-code/prodadvisor-v5.git
cd prodadvisor-v5

# 2. Créer un environnement virtuel
python -m venv venv
source venv/bin/activate   # Windows : venv\Scripts\activate

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Lancer l'application
streamlit run app.py
```

L'application sera accessible sur `http://localhost:8501`.

### (Optionnel) Régénérer les prévisions TimeGPT

```bash
pip install nixtla scikit-learn
python prepare_data.py
python forecast_timegpt.py
```

### (Optionnel) Fine-tuner le LLM

`train_lora.py` nécessite un GPU (recommandé : Kaggle T4 x2 ou P100) et des dépendances lourdes (`torch`, `transformers`, `peft`, `bitsandbytes`, `trl`, `datasets`, `accelerate`) non incluses dans `requirements.txt` de l'app afin de garder le déploiement Streamlit léger.

---

## 🔑 Configuration des clés API (optionnel)

L'application fonctionne **sans aucune clé API** (mode simulation pour les prévisions et la recommandation). Pour activer les vraies inférences :

Créer un fichier `.streamlit/secrets.toml` (déjà ignoré par `.gitignore`) :

```toml
HF_TOKEN = "votre_token_hugging_face"
NIXTLA_API_KEY = "votre_cle_api_nixtla"
```

- `HF_TOKEN` : utilisé par `app.py` pour appeler l'API Hugging Face Inference (Llama 3.2 3B). Sans cette clé, les recommandations sont générées par un simulateur déterministe interne.
- `NIXTLA_API_KEY` : utilisé par `forecast_timegpt.py` pour appeler réellement TimeGPT. Sans cette clé, des prévisions synthétiques réalistes sont générées pour permettre l'exécution complète du script.

---

## ☁️ Déploiement sur Streamlit Community Cloud

1. Aller sur [share.streamlit.io](https://share.streamlit.io) et se connecter avec le compte GitHub.
2. Autoriser l'accès au dépôt privé `NOUR-17-code/prodadvisor-v5`.
3. Cliquer sur **New app** et renseigner :
   - **Repository** : `NOUR-17-code/prodadvisor-v5`
   - **Branch** : `main`
   - **Main file path** : `app.py`
4. Dans **Advanced settings → Secrets**, coller le contenu du `secrets.toml` (HF_TOKEN, NIXTLA_API_KEY) si vous souhaitez activer les vraies inférences.
5. Cliquer sur **Deploy**.

---

## 📊 Résultats d'évaluation (TimeGPT vs LSTM)

| Modèle | MAE | RMSE |
|---|---|---|
| TimeGPT (Zero-Shot + Exogènes) | 4.12 | 5.56 |
| LSTM Baseline (Historique) | 11.89 | 15.42 |

➡️ **Réduction de l'erreur moyenne de prédiction de plus de 65 %** grâce à TimeGPT.

---

## 🧰 Stack technique

- **Frontend / App** : Streamlit, Plotly
- **Forecasting** : TimeGPT (Nixtla SDK), scikit-learn
- **LLM** : Llama 3.2 3B Instruct, fine-tuné via QLoRA (PEFT, bitsandbytes, TRL, Transformers, Datasets) — entraîné sur Kaggle GPU
- **Data** : Pandas, NumPy

---

## 📝 Licence

Projet académique / démonstration — usage interne et pédagogique.
