# HybridIoT2

Système multi-agent coopératif pour l'estimation de données manquantes de capteurs météorologiques en environnement urbain (Toulouse).

Développé dans le cadre d'un stage de recherche à l'IRIT, équipe SMAC.

## Structure

- `hybridiot/` — package principal
  - `agent.py` — agent de capteur (dataset local, détection de fautes, méthodes d'estimation)
  - `sensor.py` — capteur (id, position, is_virtual)
  - `environment.py` — environnement partagé, canal de communication inter-agents
  - `estimator.py` — helper de calcul pur pour l'estimation de valeurs manquantes
  - `data_window.py` — fenêtre de données
  - `base_dataset.py` — classe abstraite HybridIoTDataset
  - `meteo_dataset.py` — dataset météo (chargement CSV, requêtes)
  - `position.py` — classe de position géographique
- `notebooks/` — notebooks de test et d'analyse
- `data/` — fichiers CSV (données météo Toulouse) et résultats sauvegardés
- `scripts/` — scripts utilitaires

## Architecture

Les agents ne communiquent jamais directement : tous les échanges sont médiés par l'`Environment`, qui sélectionne lui-même les collaborateurs (l'agent demandeur ne les choisit pas).

La séparation des responsabilités est stricte :
- l'`Agent` prépare les données depuis son dataset local (construction des ACW, collecte via l'Environment) ;
- l'`Estimator` effectue uniquement les calculs — il reçoit des `List[float]` et retourne un `float`, sans aucune référence à `Agent` ni à `Environment`.

Plusieurs instances d'`Environment` peuvent coexister : un environnement global et, à terme, des environnements locaux (zones de confiance). `register_agent()` revendique l'agent et fixe sa référence — réservé à l'environnement global ; `add_agent()` insère un agent par référence sans modifier `agent.environment` — pour les environnements locaux.

## Méthodes d'estimation

Trois méthodes sont prévues par le modèle, mutuellement exclusives quant au choix des collaborateurs :

1. **Endogène non coopérative** (`estimate_from_history`) — l'agent utilise son propre historique : il compare sa fenêtre courante aux ACW passées, retient les 10 plus similaires et en dérive un poids.
2. **Zone de confiance** (endogène coopérative) — pas encore implementé.
3. **Exogène coopérative** (`estimate_exogenous`) — l'agent transmet Σ, les instants de ses ACW les plus similaires ; chaque collaborateur hétérogène compare ses propres ACW à ces instants et retourne un **poids** (une variation, jamais une valeur) ; l'agent combine par histogramme avec son estimation endogène.

Les méthodes ne modifient pas le dataset : l'appelant décide d'ajouter l'estimation via `add_value(..., is_estimate=True)`, ce qui trace le timestamp dans `estimated_timestamps`.

Taille des ACW fixée à 8, fenêtres consécutives non chevauchantes. La taille variable (§5.10 de la thèse, fondée sur l'écart-type entre échantillons) dépend du mécanisme de fréquence d'acquisition adaptative, hors périmètre.

## Installation

```bash
pip install -r requirements.txt
```

## Téléchargement des données

Le script `scripts/download_toulouse_meteo.py` télécharge les données météo depuis Open Data Toulouse Métropole. Les instructions détaillées sont documentées en tête du fichier.

```bash
python scripts/download_toulouse_meteo.py --config archive_2020_full
python scripts/download_toulouse_meteo.py --search "montaudran"
python scripts/download_toulouse_meteo.py --list
```

## Utilisation

```python
from hybridiot.meteo_dataset import MeteoDataset
from hybridiot.environment import Environment
from hybridiot.agent import Agent

# Charger un dataset
ds = MeteoDataset.from_csv("data/toulouse_meteo_2020.csv")
env = Environment(base_dataset=ds)

# Un agent par station et par type de donnée
agent = Agent(id="Castelginest Ecole", position=pos, data_type="temperature")
env.register_agent(agent)
agent.load_from_dataset(ds)

# Estimation par historique
estimation = agent.estimate_from_history(timestamp)

# Estimation exogène coopérative (30 % des collaborateurs hétérogènes)
estimation = agent.estimate_exogenous(timestamp, collab_ratio=0.3)

# L'appelant décide d'ajouter l'estimation au dataset
agent.add_value(estimation, timestamp, is_estimate=True)
```

## Notebooks

- `test_meteo_dataset.ipynb` — chargement et exploration du dataset
- `test_agent.ipynb` — agent, dataset local, détection de fautes
- `test_agent_environment.ipynb` — enregistrement et communication médiée
- `test_references_environnements.ipynb` — sémantique des références Python entre environnements multiples
- `test_estimate_from_history.ipynb` — validation de la méthode par historique
- `analyse_finale.ipynb` — évaluation complète des deux méthodes sur les 18 agents

## Données

Le dataset principal couvre l'année 2020 : 6 stations × 3 types de données (température, humidité, pression), au pas de 15 minutes, soit 18 agents.

Attention : le CSV utilise la valeur sentinelle `-50` comme marqueur de donnée manquante en température. Ces valeurs sont filtrées par `_is_valid_value()` via `METEO_VALID_RANGES`.

Un second fichier (données live de janvier 2023, 5 stations) est fourni pour les tests. Pour télécharger d'autres datasets, utiliser le script de téléchargement.