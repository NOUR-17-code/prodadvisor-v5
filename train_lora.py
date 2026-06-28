import os
import torch
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

# =====================================================================
# CONFIGURATION & PARAMÈTRES
# =====================================================================
MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"  # Requiert acceptation des conditions Meta sur Hugging Face
# Alternative publique libre : "unsloth/Llama-3.2-3B-Instruct" (souvent utilisé sur Kaggle pour le prototypage)
OUTPUT_DIR = "./prodadvisor_llama_adapters"
DATASET_PATH = "fashion_boutique_cleaned.csv"

def print_gpu_info():
    """Affiche les informations GPU disponibles sur Kaggle (T4 x2 ou P100)"""
    print("=== ÉTAPE 3 : Pipeline de Fine-Tuning QLoRA sur Kaggle ===")
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        print(f"GPU(s) détecté(s) : {device_count}")
        for i in range(device_count):
            print(f"  - GPU {i} : {torch.cuda.get_device_name(i)}")
    else:
        print("[WARNING] Aucun GPU détecté. L'entraînement QLoRA nécessite un environnement GPU (Kaggle T4 x2/P100).")

def format_instruction_prompt(row):
    """
    Formate le prompt d'instruction pour Llama 3.2 en fusionnant :
    - Filtres (Season, Category)
    - Demande prédite numérique issue de TimeGPT (simulée ici par recommended_quantity)
    - Contexte d'origine et la réponse cible.
    """
    # Contexte d'alerte sur les retours (Risk Analytics simulé en entraînement)
    # Dans le dataset propre, on peut déduire un taux de retour moyen ou utiliser les colonnes existantes
    return_info = ""
    if row.get('is_returned', False) == True or str(row.get('is_returned')).lower() == 'true':
        return_info = f"\n- Alerte risque : Ce produit a été retourné. Cause fréquente : {row.get('return_reason', 'Non spécifiée')}."

    system_message = (
        "Tu es l'expert MLOps et Conseiller Commercial senior de PRODADVISOR. "
        "Ta mission est de rédiger une recommandation de stock claire, concise et exploitable "
        "en te basant sur la demande prévue par TimeGPT, les filtres produits et l'historique des retours."
    )
    
    user_message = (
        f"Données d'analyse :\n"
        f"- Catégorie : {row['category']}\n"
        f"- Saison : {row['season']}\n"
        f"- Marque : {row['brand']}\n"
        f"- Couleur : {row['color']}\n"
        f"- Prix de base : {row['original_price']} €\n"
        f"- Demande prévue par TimeGPT : {row['recommended_quantity']} unités\n"
        f"- Taux de démarque appliqué : {row['markdown_percentage']}%\n"
        f"{return_info}\n\n"
        f"Génère la recommandation de stock stratégique en français."
    )
    
    # Structure du prompt au format Llama 3.2 Instruct
    prompt = (
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_message}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n\n{user_message}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n{row['response']}<|eot_id|>"
    )
    return {"text": prompt}

def run_qlora_finetuning():
    print_gpu_info()
    
    # 1. Chargement et formatage du dataset
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Le fichier {DATASET_PATH} est introuvable. Assurez-vous d'avoir le CSV dans votre répertoire Kaggle.")
        
    df = pd.read_csv(DATASET_PATH)
    print(f"Chargement de {len(df)} lignes de données d'entraînement.")
    
    # Création du dataset Hugging Face et application du template de prompt
    dataset = Dataset.from_pandas(df)
    dataset = dataset.map(format_instruction_prompt, remove_columns=dataset.column_names)
    print("Exemple de prompt formaté :")
    print(dataset[0]['text'][:500] + "\n...")

    # 2. Configuration de la quantification 4-bit (BitsAndBytes)
    # Permet de charger Llama 3.2 3B avec seulement ~2.5 Go de VRAM (idéal pour GPU T4 x2)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )

    # 3. Chargement du Tokenizer et du Modèle
    print(f"Chargement du tokenizer pour {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print(f"Chargement du modèle base {MODEL_NAME} en 4-bit...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    
    # Préparation du modèle pour l'entraînement avec quantification
    model = prepare_model_for_kbit_training(model)

    # 4. Configuration de LoRA
    # Ciblage des modules d'attention de Llama 3.2 3B
    lora_config = LoraConfig(
        r=16,                           # Dimension de la matrice d'adaptation
        lora_alpha=32,                  # Facteur d'échelle
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj", 
            "gate_proj", "up_proj", "down_proj"
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    print("Application des configurations LoRA au modèle...")
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 5. Arguments d'entraînement optimisés pour Kaggle
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        logging_steps=10,
        max_steps=100,                  # Ajustez les étapes en fonction du temps d'entraînement voulu (ex: 500-1000 pour convergence)
        optim="paged_adamw_8bit",       # Optimiseur mémoire optimisé
        fp16=True,                      # Utilisation de la précision mixte 16-bit
        save_strategy="steps",
        save_steps=50,
        evaluation_strategy="no",       # Pas d'évaluation coûteuse pendant le run rapide
        report_to="none"                # Pas de tracking externe WandB obligatoire
    )

    # 6. SFTTrainer pour l'entraînement supervisé
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=512,             # Longueur max des prompts
        tokenizer=tokenizer,
        args=training_args,
        peft_config=lora_config,
    )

    print("Lancement de l'entraînement QLoRA...")
    trainer.train()
    
    # 7. Sauvegarde des adaptateurs LoRA
    print(f"Entraînement terminé ! Sauvegarde des adaptateurs LoRA dans {OUTPUT_DIR}...")
    trainer.model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("Sauvegarde effectuée. Les adaptateurs sont prêts à être téléchargés ou déployés dans Streamlit !")

if __name__ == '__main__':
    # Pour s'exécuter sur Kaggle, assurez-vous d'avoir installé les packages requis :
    # !pip install -q transformers peft bitsandbytes trl datasets accelerate
    run_qlora_finetuning()
