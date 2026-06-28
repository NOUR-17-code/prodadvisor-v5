import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

def prepare_time_series(csv_path='fashion_boutique_cleaned.csv', target_freq='W'):
    """
    Charge le dataset de la boutique de mode et l'agrège par pas de temps (hebdomadaire par défaut)
    pour correspondre au format attendu par TimeGPT (unique_id, ds, y) tout en conservant
    les variables exogènes moyennes (markdown_percentage, current_price).
    
    Args:
        csv_path (str): Chemin vers le fichier CSV propre.
        target_freq (str): Fréquence d'agrégation ('W' pour hebdomadaire, 'M' pour mensuel).
        
    Returns:
        tuple: DataFrames train et test prêts pour l'inférence.
    """
    print("=== ÉTAPE 1 : Chargement et Préparation du Dataset ===")
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Le fichier {csv_path} est introuvable. Veuillez vérifier son emplacement.")
        
    # 1. Chargement des données
    df = pd.read_csv(csv_path)
    print(f"Dataset chargé avec succès. Dimensions : {df.shape}")
    
    # 2. Conversion de la colonne date
    df['purchase_date'] = pd.to_datetime(df['purchase_date'])
    
    # 3. Agrégation par unique_id (category) et ds (date resamplée)
    # unique_id: la série temporelle est modélisée par catégorie de vêtements
    # ds: la date de début de période (semaine ou mois)
    # y: la demande (volume cumulé des ventes représenté par la somme de recommended_quantity)
    # exogènes: moyenne du taux de démarque (markdown_percentage) et du prix final (current_price)
    
    # On crée une colonne temporelle de regroupement selon la fréquence choisie
    df['ds'] = df['purchase_date'].dt.to_period(target_freq).dt.to_timestamp()
    
    # Groupement des données
    aggregated_df = df.groupby(['category', 'ds']).agg({
        'recommended_quantity': 'sum',            # Variable cible (demande / volume de vente)
        'markdown_percentage': 'mean',            # Exogène 1 : Taux de promotion moyen
        'current_price': 'mean',                  # Exogène 2 : Prix de vente moyen
    }).reset_index()
    
    # Renommer les colonnes pour se conformer au standard Nixtla/TimeGPT
    aggregated_df = aggregated_df.rename(columns={
        'category': 'unique_id',
        'recommended_quantity': 'y'
    })
    
    # Tri par date pour garantir l'ordre chronologique
    aggregated_df = aggregated_df.sort_values(by=['unique_id', 'ds']).reset_index(drop=True)
    
    print(f"Données agrégées ({target_freq}) : {aggregated_df.shape} lignes.")
    print(f"Catégories identifiées : {aggregated_df['unique_id'].unique().tolist()}")
    
    # 4. Division Chronologique Train / Test (sans fuite de données futures)
    # On prend les 80% premières dates pour chaque série comme Train, et les 20% restantes comme Test.
    train_dfs = []
    test_dfs = []
    
    for cat in aggregated_df['unique_id'].unique():
        cat_df = aggregated_df[aggregated_df['unique_id'] == cat]
        split_idx = int(len(cat_df) * 0.8)
        
        train_dfs.append(cat_df.iloc[:split_idx])
        test_dfs.append(cat_df.iloc[split_idx:])
        
    df_train = pd.concat(train_dfs).reset_index(drop=True)
    df_test = pd.concat(test_dfs).reset_index(drop=True)
    
    print(f"Train set : {df_train.shape[0]} observations (du {df_train['ds'].min().strftime('%Y-%m-%d')} au {df_train['ds'].max().strftime('%Y-%m-%d')})")
    print(f"Test set : {df_test.shape[0]} observations (du {df_test['ds'].min().strftime('%Y-%m-%d')} au {df_test['ds'].max().strftime('%Y-%m-%d')})")
    
    # Sauvegarde des fichiers intermédiaires
    df_train.to_csv('train_time_series.csv', index=False)
    df_test.to_csv('test_time_series.csv', index=False)
    print("Fichiers 'train_time_series.csv' et 'test_time_series.csv' sauvegardés avec succès.\n")
    
    return df_train, df_test

if __name__ == '__main__':
    df_train, df_test = prepare_time_series()
