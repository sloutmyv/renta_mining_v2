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
- **Ligne de moyenne** : affichage en vert gras de la trajectoire moyenne attendue.
- **Métriques en temps réel** : affichage du prix actuel du BTC et du prix cible simulé, centré verticalement.
- Paramètres personnalisables : période historique, période de prédiction, nombre de simulations, évolution cible, volatilité (calculée automatiquement par défaut).

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
