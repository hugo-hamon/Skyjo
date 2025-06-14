# SkyjoEnv

Environnement pour le jeu de cartes **Skyjo**, dans le but d'entrainer des agents d'apprentissage par renforcement.

## Fonctionnalités principales

- API compatible `gym`: méthodes `reset`, `step`, `render`, etc.
- Multi-joueurs (2-8 joueurs)
- Gestion complète du tour par tour, pioche, remplacement, retournement, fin de partie
- Tests unitaires (Pytest)

## Structure

```
Skyjo/
├── src/
│   └── skyjo_env.py # Environnement principal
├── test/
│   └── test_env.py # Tests pour l'environnement
└── example/
    └── example.py # Example d'utilisation de l'environnement
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate # sur Windows : .venv\Scripts\activate
pip install uv
uv sync
uv run pytest # pour vérifier que tout fonctionne
```

## Exemple d'utilisation

```bash
uv run example/example.py
```