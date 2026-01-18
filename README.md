# Renta Mining V2

Simulateur industriel complet de rentabilité et ROI pour le minage de Bitcoin.

## Fonctionnalités Clés
- **Simulations Monte Carlo Multi-niveaux** : Prix du BTC, Difficulté de minage, et Valeur des récompenses en USD/XPF.
- **Analyse de Rentabilité Triple** :
    - **Revenu Théorique** : Basé sur le hashrate et la difficulté.
    - **Revenu Brut** : Intègre les frais de Pool et le taux de rejet.
    - **Profit Net** : Déduit l'OPEX électrique (ajusté par l'Uptime et le PUE).
- **Gestion Financière Complète (XPF/EUR)** :
    - Saisie des **CAPEX** (Machines, Infrastructure, Ingénierie) en Euros.
    - Saisie des **OPEX Fixes** (Maintenance, Assurance, Supervision) en Euros.
    - Conversion automatique en **Franc Pacifique (XPF)**.
- **Analyse de ROI** : Graphique de flux de trésorerie cumulé avec estimation automatique de la date du **Point Mort**.
- **Visualisation des Coûts** : Pie Chart de répartition des charges mensuelles (Élec vs Fixes).

## Architecture de l'Application

L'application Streamlit est structurée en 7 lignes synchronisées pour une lecture fluide du "funnel" de minage :

1. **Cours BTC/USD** : Historique et prédictions MC (Thème bleu).
2. **Difficulté de Minage** : Modèle prédictif basé sur le prix avec inertie et stochastique (Thème orange).
3. **Récompenses par Bloc** : Visualisation des Halvings et valeur en USD (Thème rouge/cyan).
4. **Revenu Théorique** : Potentiel de gain pur en BTC & XPF (Thème violet).
5. **Revenu Brut** : Revenu net de frais de pool et rejets (Thème orangé/or).
6. **Profit Net** : Revenu après déduction de l'OPEX électrique (Thème vert).
7. **ROI & Cash Flow** : Courbe de retour sur investissement partant du CAPEX initial (Thème blanc/citron).

### Innovations Techniques
- **Timezone UTC** : Standardisation de toutes les sources de données (yfinance, blockchain.com) pour un alignement parfait au point "Maintenant".
- **Difficulté Stochastique** : Intégration d'un bruit aléatoire quotidien pour simuler la réalité du réseau dès le premier jour de prédiction.
- **Double Axe Y** : Tous les graphiques financiers affichent simultanément le BTC et sa contrevaleur en XPF.
- **Opacité Dynamique** : Nuages Monte Carlo renforcés (alpha 0.12) pour une meilleure lecture des scénarios extrêmes.

## Utilisation

### Lancement
```bash
streamlit run app.py
```

### Paramètres Principaux
- **Hardware** : Nombre de machines, Hashrate, Puissance (W).
- **Réseau** : Frais de transaction moyens par bloc.
- **OPEX** : Coût du kWh (XPF), Uptime (%), PUE.
- **Financier** : CAPEX global et charges fixes mensuelles.

---
Développement et maintenance via **Antigravity**.
