# Renta Mining V2

Bitcoin mining profitability and ROI simulator.

## Features
- Real-time mining profitability calculation.
- Net Profit and ROI simulation.
- XPF to USD conversion for electricity costs.
- CAPEX and OPEX integration.

## Streamlit Application: Evaluation rentabilité projet

Une application interactive pour visualiser le cours du BTC/USD et explorer des scénarios futurs via des simulations de Monte Carlo.

### Fonctionnalités
- Graphique interactif du cours historique du BTC.
- 100 simulations Monte Carlo (ajustables) calculées sur la base de la volatilité et d'un objectif de prix.
- **Visualisation des Récompenses par Bloc** :
    - Graphique à double axe Y (BTC et USD).
    - Simulation Monte Carlo de la valeur future du bloc en USD (basée sur le prix simulé).
    - Tracé en "escalier" (stairs) pour refléter fidèlement l'impact discret des Halvings.
- **Modèle de Difficulté Avancé** :
    - Gestion de l'inertie (lag) pour refléter le délai de déploiement du hashrate.
    - Ajout d'une composante stochastique (bruit aléatoire) pour une divergence immédiate des trajectoires.
- **Interface Utilisateur Premium** :
    - Mise en page par lignes avec alignement parfait des zones de tracé.
    - Indicateur visuel "Aujourd'hui" (ligne verticale pointillée) sur tous les graphiques.
    - Thématique de couleurs harmonisée (bleu pour le prix, orange pour la difficulté, rouge/cyan pour les récompenses).
    - Métriques minimalistes alignées verticalement.
- **Interface Intuitive** : Les paramètres sont en haut, suivis de deux lignes (Prix et Difficulté). Chaque ligne contient son propre graphique et ses métriques (Actuel/Cible) à droite.
- **Légendes Optimisées** : Les légendes sont placées à l'intérieur des graphiques en haut à gauche.
- **Paramètres de Difficulté** : Réglage de la sensibilité et de l'inertie (en jours) en haut de la page.

- **Simulation de Difficulté** : Ajout d'un second graphique montrant l'historique et la prédiction de la difficulté de minage.
- **Modèle avec Inertie** : La difficulté suit l'évolution du prix du BTC avec une sensibilité et un retard (inertie) paramétrables.
- **Nuage Monte Carlo** : Visualisation de l'incertitude sur la difficulté future.
- **Paramètres de Difficulté** : Réglage de la sensibilité et de l'inertie (en jours).

### Comment lancer l'application
```bash
streamlit run app.py
```

## Setup Instructions

### Virtual Environment
This project uses a Python virtual environment to manage dependencies.

1. **Active the environment**:
   ```bash
   source venv/bin/activate
   ```

2. **Install dependencies** (when available):
   ```bash
   pip install -r requirements.txt
   ```

## Development
Created and maintained with Antigravity.
