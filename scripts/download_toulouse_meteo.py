"""
Script de téléchargement des données météo - Open Data Toulouse Métropole

    Fonctionnalités principales :
    - Récupération automatique des coordonnées des stations
      (latitude, longitude, altitude) via le dataset de métadonnées
      "stations-meteo-en-place"
    - Configurations prédéfinies directement dans le script via CONFIGS
      (pas besoin de fichier externe)
    - Compatibilité avec plusieurs années et différents types de datasets
      sans modifier le code
    - Filtrage temporel côté serveur via ODSQL
      (date_debut / date_fin)
    - Recherche de datasets OpenData par mot-clé
    - Inspection des colonnes disponibles dans un dataset

UTILISATION NORMALE (téléchargement) :
    python download_toulouse_meteo.py
    python download_toulouse_meteo.py --config archive_2020(35.000)
    python download_toulouse_meteo.py --config archive_2020_training
    python download_toulouse_meteo.py --config live_janvier_2023
Sans --config, utilise la config par défaut : DEFAULT_CONFIG

DÉCOUVERTE DES DATASETS (mode recherche) :
    python download_toulouse_meteo.py --search "montaudran"
    python download_toulouse_meteo.py --search "archive-2020"
    python download_toulouse_meteo.py --search "station-meteo"
Parcourt tout le catalogue et filtre par mot-clé dans l'ID du dataset.
Utile pour trouver les IDs à mettre dans CONFIGS sans aller sur le portail web.

INSPECTION DES COLONNES D'UN DATASET :
    python download_toulouse_meteo.py --columns "08-station-meteo-toulouse-basso-cambo-archive-2020"
    python download_toulouse_meteo.py --columns "07-station-meteo-toulouse-avenue-de-grande-bretagne" 
Affiche le nom et le type de chaque colonne disponible dans le dataset.
Cette fonctionnalité est utile pour :
    - identifier la colonne timestamp ;
    - identifier les colonnes de données à conserver ;
    - détecter les différences de structure entre datasets ;
    - créer une nouvelle configuration dans CONFIGS.

LISTE DES CONFIGS DISPONIBLES :
    python download_toulouse_meteo.py --list

FORMAT DES DATES : "YYYY-MM-DD" ou "YYYY-MM-DDTHH:MM:SS"
  date_debut=None  ->  pas de borne inférieure
  date_fin=None    ->  pas de borne supérieure
  Les deux à None  ->  téléchargement complet du dataset
"""

import argparse
import io
import os
import sys

import pandas as pd
import requests


# CONFIGURATIONS 

CONFIGS = {
    # Archive 2020 avec 35.000 lignes (6 stations, année complète) 
    "archive_2020(35.000)": {
        "description": "6 stations, année 2020 complète",
        "output_file": "toulouse_meteo_2020(35.000).csv",
        "date_debut":  None,
        "date_fin":    None,
        "timestamp_col": "dd_mm_yy_hh_mm_utc",
        "columns": [
            "temperature",
            "humidite",
            "pression",
            "pluie",
            "pluie_intensite_max",
        ],
        "station_ids": [
            {"id": "05-station-meteo-toulouse-nakache-archive2019"},
            {"id": "07-station-meteo-toulouse-avenue-de-grande-bretagne-archive-2020"},
            {"id": "08-station-meteo-toulouse-basso-cambo-archive-2020"},
            {"id": "10-station-meteo-castelginest-ecole-archive-2020"},
            {"id": "19-station-meteo-mondouzil-mairie-archive-2020"},
            {"id": "22-station-meteo-colomiers-za-perget-archive-2020"}
        ],
    },

    # Example pour un training set: fevrier-decembre 2020 
    "archive_2020_training": {
        "description": "6 stations, 2020 sans janvier (training)",
        "output_file": "toulouse_meteo_2020_training.csv",
        "date_debut":  "2020-02-01",
        "date_fin":    "2020-12-31",
        "timestamp_col": "dd_mm_yy_hh_mm_utc",
        "columns": [
            "temperature",
            "humidite",
            "pression",
            "pluie",
            "pluie_intensite_max",
        ],

        "station_ids": [
            {"id": "05-station-meteo-toulouse-nakache-archive2019"},
            {"id": "07-station-meteo-toulouse-avenue-de-grande-bretagne-archive-2020"},
            {"id": "08-station-meteo-toulouse-basso-cambo-archive-2020"},
            {"id": "10-station-meteo-castelginest-ecole-archive-2020"},
            {"id": "19-station-meteo-mondouzil-mairie-archive-2020"},
            {"id": "22-station-meteo-colomiers-za-perget-archive-2020"}
        ],
    },

    # Example pour un validation set: janvier 2020 
    "archive_2020_validation": {
        "description": "6 stations, seul janvier 2020 (validation)",
        "output_file": "toulouse_meteo_2020_validation.csv",
        "date_debut":  "2020-01-01",
        "date_fin":    "2020-01-31",
        "timestamp_col": "dd_mm_yy_hh_mm_utc",
        "columns": [
            "temperature",
            "humidite",
            "pression",
            "pluie",
            "pluie_intensite_max",
        ],
        "station_ids": [
            {"id": "05-station-meteo-toulouse-nakache-archive2019"},
            {"id": "07-station-meteo-toulouse-avenue-de-grande-bretagne-archive-2020"},
            {"id": "08-station-meteo-toulouse-basso-cambo-archive-2020"},
            {"id": "10-station-meteo-castelginest-ecole-archive-2020"},
            {"id": "19-station-meteo-mondouzil-mairie-archive-2020"},
            {"id": "22-station-meteo-colomiers-za-perget-archive-2020"}
        ],
    },

    # Example pour un dataset live: janvier 2023 
    "live_janvier_2023": {
        "description": "6 stations live, seul janvier 2023",
        "output_file": "toulouse_meteo_live_jan2023(5stations).csv",
        "date_debut":  "2023-01-01",
        "date_fin":    "2023-01-31",
        "timestamp_col": "heure_de_paris",
        "columns": [
            "temperature",
            "humidite",
            "pression",
            "pluie",
            "pluie_intensite_max",
        ],
        "station_ids": [
            {"id": "07-station-meteo-toulouse-avenue-de-grande-bretagne", "timestamp_col": "heure_de_paris"},
            {"id": "08-station-meteo-toulouse-basso-cambo", "timestamp_col": "heure_de_paris"},
            {"id": "11-station-meteo-toulouse-soupetard", "timestamp_col": "heure_utc"},
            {"id": "12-station-meteo-toulouse-montaudran", "timestamp_col": "heure_de_paris"},
            {"id": "38-station-meteo-toulouse-parc-jardin-des-plantes", "timestamp_col": "mm_dd_yy_hh_mm"}
        ],
    },
}

# Config utilisée si --config n'est pas spécifié
DEFAULT_CONFIG = "archive_2020(35.000)"


# CONSTANTES
BASE_URL    = "https://data.toulouse-metropole.fr/api/explore/v2.1/catalog/datasets"
METADATA_ID = "stations-meteo-en-place"


# Noms alternatifs pour chaque colonne de données.
# Le script renomme automatiquement vers le nom standard avant d'agréger.
COLUMN_ALIASES: dict[str, list[str]] = {
    "temperature":         ["temperature", "temperature_en_degre_c"],
    "humidite":            ["humidite"],
    "pression":            ["pression"],
    "pluie":               ["pluie"],
    "pluie_intensite_max": ["pluie_intensite_max"],
}


# Dècouverture des datasets disponibles et inspection des colonnes

def fetch_all_dataset_ids() -> list:
    """
    Récupère tous les datasets du catalogue en paginant par tranches de 100 (limite max de l'API OpenData).
    """
    all_datasets = []
    offset       = 0
    page_size    = 100
 
    while True:
        resp = requests.get(
            BASE_URL,
            params={"limit": page_size, "offset": offset, "order_by": "dataset_id"},
            timeout=30,
        )
        resp.raise_for_status()
        batch = resp.json().get("results", [])
        all_datasets.extend(batch)
        if len(batch) < page_size:
            break
        offset += page_size
 
    return all_datasets

 
def search_datasets(keyword: str, max_results: int = 100) -> None:
    """
    Affiche les datasets dont l'ID contient le mot-clé.
    Parcourt tout le catalogue (pagination automatique) puis filtre
    côté client, la recherche porte exactement sur l'ID du dataset.
 
    Exemples:
        search_datasets("montaudran") = toutes les années de Montaudran
        search_datasets("archive-2020") = tous les datasets archives 2020
        search_datasets("station-meteo") = tous les datasets météo
    """
    print(f"\nRecherche : '{keyword}'")
    print("  Parcours du catalogue en cours...")
 
    all_datasets  = fetch_all_dataset_ids()
    keyword_lower = keyword.lower()
 
    matches = [
        d for d in all_datasets
        if keyword_lower in d.get("dataset_id", "").lower()
    ]
 
    if not matches:
        print(f"  Aucun dataset trouvé pour '{keyword}'.")
        return
 
    shown  = matches[:max_results]
    suffix = f" (affichage limité à {max_results})" if len(matches) > max_results else ""
    print(f"  {len(matches)} dataset(s) trouvé(s){suffix} :\n")
    print(f"  {'ID':<65}  {'Enregistrements':>16}")
    print("  " + "-" * 85)
 
    for item in shown:
        dataset_id = item.get("dataset_id", "?")
        n_records  = item.get("metas", {}).get("default", {}).get("records_count", "?")
        print(f"  {dataset_id:<65}  {str(n_records):>16}")
 
    print(f"\n  Copiez les IDs souhaités dans 'station_ids' de CONFIGS.")

 
def show_columns(dataset_id: str) -> None:
    """
    Affiche le nom et le type de chaque colonne disponible dans un dataset.
    À utiliser avant de créer une nouvelle config pour identifier :
      - la colonne timestamp  -> à mettre dans "timestamp_col"
      - les colonnes de données utiles -> à mettre dans "columns"
 
    Exemple: show_columns("08-station-meteo-toulouse-basso-cambo-archive-2020")
    """
    print(f"\nColonnes disponibles pour '{dataset_id}' :\n")
 
    resp = requests.get(f"{BASE_URL}/{dataset_id}", timeout=30)
    resp.raise_for_status()
    fields = resp.json().get("fields", [])
 
    if not fields:
        print("  Aucun champ trouvé (dataset_id incorrect ?).")
        return
 
    print(f"  {'Nom':<45}  {'Type':<15}")
    print("  " + "-" * 63)
    for field in fields:
        name      = field.get("name", "?")
        ftype     = field.get("type", "?")
        print(f"  {name:<45}  {ftype:<15}")
 
    print(f"\n  Workflow :")
    print(f"  1. Copiez le nom de la colonne datetime dans 'timestamp_col' de votre config.")
    print(f"  2. Copiez les noms des colonnes de données dans 'columns' de votre config.")


# Coordonnées des stations météo (latitude, longitude, altitude) via le dataset de métadonnées

def fetch_station_metadata() -> dict:
    """
    Télécharge 'stations-meteo-en-place' et retourne un dictionnaire :
        { "08-station-meteo-...": {"lat": ..., "lon": ..., "altitude": ...} }
    """
    print("\n[Métadonnées] Récupération des coordonnées...")
    url  = f"{BASE_URL}/{METADATA_ID}/exports/csv"
    resp = requests.get(url, params={"delimiter": ";"}, timeout=60)
    resp.raise_for_status()

    df_meta = pd.read_csv(io.StringIO(resp.text), sep=";", low_memory=False)
    print(f"  -> {len(df_meta)} stations trouvées")

    required = {"id_nom", "latitude", "longitude", "altitude"}
    missing  = required - set(df_meta.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes dans les métadonnées : {missing}")

    metadata = {}
    for _, row in df_meta.iterrows():
        sid = str(row["id_nom"]).strip()
        metadata[sid] = {
            "lat":      float(row["latitude"]),
            "lon":      float(row["longitude"]),
            "altitude": float(row["altitude"]),
        }
    return metadata


def match_metadata(dataset_id: str, metadata: dict) -> dict:
    """
    Associe un dataset_id à ses coordonnées par le préfixe numérique.
    "08-station-meteo-toulouse-basso-cambo-archive-2020" -> cherche "08-..."
    """
    station_number = dataset_id.split("-")[0]
    for meta_id, coords in metadata.items():
        if meta_id.startswith(station_number + "-"):
            return coords
    raise KeyError(
        f"Aucune métadonnée pour la station n°{station_number} "
        f"(dataset_id='{dataset_id}')"
    )



# Téléchargement des données météo pour une station ou un ensemble de stations

def build_date_filter(ts_col: str, date_debut, date_fin) -> str:
    """
    Construit le filtre ODSQL pour les dates.
    Les deux dates à None -> pas de filtre (téléchargement complet).
    Exemples de résultats :
        "dd_mm_yy_hh_mm_utc >= '2020-02-01' AND dd_mm_yy_hh_mm_utc <= '2020-12-31'"
        "dd_mm_yy_hh_mm_utc >= '2023-01-01'"
        None
    """
    parts = []
    if date_debut:
        parts.append(f"{ts_col} >= '{date_debut}'")
    if date_fin:
        parts.append(f"{ts_col} <= '{date_fin}'")
    return " AND ".join(parts) if parts else None


def get_record_count(dataset_id: str, date_filter) -> int:
    """Retourne le nombre de records correspondant au filtre (ou au total)."""
    params = {"select": "count(*)", "limit": 1}
    if date_filter:
        params["where"] = date_filter
    resp = requests.get(f"{BASE_URL}/{dataset_id}/records", params=params, timeout=30)
    resp.raise_for_status()
    return resp.json().get("total_count", 0)


def download_station(station_entry: dict, coords: dict, config: dict) -> pd.DataFrame:
    """
    Télécharge les données d'une station:
    station_entry : dict avec au moins {"id": "..."} et optionnellement
                    {"timestamp_col": "..."} pour surcharger le défaut de config.
    coords        : coordonnées GPS de la station (lat, lon, altitude)
    config        : config globale (timestamp_col par défaut, columns, dates)
    
    La colonne timestamp est renommée en "timestamp" dans le CSV de sortie,
    quel que soit son nom d'origine.
    
    Si date_filter est fourni, il est transmis au serveur via le paramètre
    'where' -> seules les lignes correspondantes sont téléchargées.
    """
    # Nom lisible depuis l'ID
    dataset_id   = station_entry["id"]
    parts        = dataset_id.split("-")
    name_parts   = [p for p in parts if p not in ("station", "meteo", "archive") and not p.isdigit()]
    station_name = " ".join(name_parts).title()
 
    print(f"\n[{station_name}]")
 
    # timestamp_col : valeur de la stazione en priorité, sinon défaut de la config
    ts_col  = station_entry.get("timestamp_col") or config["timestamp_col"]
    columns     = config["columns"]

    date_filter = build_date_filter(ts_col, config.get("date_debut"), config.get("date_fin"))
    if date_filter:
        print(f"  Filtre : {date_filter}")
    else:
        print(f"  Filtre : aucun (téléchargement complet)")
 
    print(f"  -> {get_record_count(dataset_id, date_filter)} enregistrements attendus")
 
    # Téléchargement
    params = {"use_labels": "false", "delimiter": ";"}
    if date_filter:
        params["where"] = date_filter
 
    resp = requests.get(f"{BASE_URL}/{dataset_id}/exports/csv", params=params, timeout=120)
    resp.raise_for_status()
 
    df = pd.read_csv(io.StringIO(resp.text), sep=";", low_memory=False)
    print(f"  -> {len(df)} lignes téléchargées")

    # Normalisation des noms de colonnes vers les noms standard.
    # Ex: 'temperature_en_degre_c' -> 'temperature'
    # Permet d'agréger des stations avec des noms de colonnes différents.
    for standard_name, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in df.columns and alias != standard_name:
                df.rename(columns={alias: standard_name}, inplace=True)
                print(f"  Colonne renommée : '{alias}' -> '{standard_name}'")
                break

    # Sélection colonnes
    missing = [c for c in columns if c not in df.columns]
    if missing:
        print(f"  Colonnes absentes : {missing}")
 
    cols_present = [c for c in columns if c in df.columns]
    ts_in_df     = [ts_col] if ts_col in df.columns else []
    df = df[ts_in_df + cols_present].copy()
 
    # Conversion timestamp + renommage unifié -> "timestamp"
    if ts_col in df.columns:
        df[ts_col] = pd.to_datetime(df[ts_col], format="ISO8601", errors="coerce")
        df.rename(columns={ts_col: "timestamp"}, inplace=True)
 
    # 5. Métadonnées station
    df["station_name"] = station_name
    df["latitude"]     = coords["lat"]
    df["longitude"]    = coords["lon"]
    df["altitude"]     = coords["altitude"]
 
    # 6. Rapport qualité
    n_null = int(df.isnull().sum().sum())
    n_temp_anomaly = int((df["temperature"] < -10).sum()) if "temperature" in df.columns else 0
    print(f"  -> Nulles : {n_null}  |  Anomalies temp. (<-10°C) : {n_temp_anomaly}")
 
    return df



def download_all(config: dict, metadata: dict) -> pd.DataFrame:
    """Télécharge toutes les stations d'une config et retourne le DataFrame agrégé."""
    frames = []
 
    for station_entry in config["station_ids"]:
        dataset_id = station_entry["id"]
        try:
            coords = match_metadata(dataset_id, metadata)
            df     = download_station(station_entry, coords, config)
            frames.append(df)
        except KeyError as e:
            print(f"\nCoordonnées manquantes : {e}")
        except requests.HTTPError as e:
            print(f"\nErreur HTTP '{dataset_id}' : {e}")
        except Exception as e:
            print(f"\nErreur inattendue '{dataset_id}' : {e}")
 
    if not frames:
        raise RuntimeError("Aucune station téléchargée avec succès.")
 
    aggregated = pd.concat(frames, ignore_index=True)
    aggregated.sort_values(["station_name", "timestamp"], inplace=True)
    aggregated.reset_index(drop=True, inplace=True)
    return aggregated




# MAIN

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Téléchargement données météo Toulouse Open Data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            Exemples :
              python download_toulouse_meteo.py
              python download_toulouse_meteo.py --config archive_2020_validation
              python download_toulouse_meteo.py --search "archive-2020"
              python download_toulouse_meteo.py --columns "08-station-meteo-toulouse-basso-cambo-archive-2020"
              python download_toulouse_meteo.py --list
        """
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG,
                        help=f"Nom de la config (défaut: {DEFAULT_CONFIG})")
    parser.add_argument("--search", metavar="MOT_CLE",
                        help="Recherche datasets par mot-clé dans l'ID")
    parser.add_argument("--columns", metavar="DATASET_ID",
                        help="Affiche les colonnes disponibles d'un dataset")
    parser.add_argument("--list", action="store_true",
                        help="Liste toutes les configurations disponibles")
    args = parser.parse_args()

    if args.list:
        print("\nConfigurations disponibles :\n")
        for name, cfg in CONFIGS.items():
            debut = cfg.get("date_debut") or "début"
            fin   = cfg.get("date_fin")   or "fin"
            print(f"  {name}")
            print(f"    {cfg['description']}")
            print(f"    Période      : {debut} -> {fin}")
            print(f"    Timestamp    : {cfg.get('timestamp_col', '?')} (défaut)")
            # Affiche les surcharges par stazione si présentes
            overrides = [s for s in cfg.get("station_ids", []) if s.get("timestamp_col")]
            if overrides:
                for s in overrides:
                    print(f"                   {s['id']} -> '{s['timestamp_col']}'")
            print(f"    Colonnes     : {cfg.get('columns', [])}")
            print(f"    Sortie       : {cfg['output_file']}\n")
        sys.exit(0)

    # Recherche de datasets par mot-clé
    if args.search:
        search_datasets(args.search)
        sys.exit(0)

    # Inspection des colonnes
    if args.columns:
        show_columns(args.columns)
        sys.exit(0)

    # Téléchargement
    if args.config not in CONFIGS:
        print(f"✗ Config '{args.config}' introuvable.")
        print(f"  Configs disponibles : {list(CONFIGS.keys())}")
        sys.exit(1)

    config = CONFIGS[args.config]

    # Vérification des clés obligatoires
    for key in ("timestamp_col", "columns", "station_ids", "output_file"):
        if key not in config:
            print(f"Clé manquante dans la config '{args.config}' : '{key}'")
            sys.exit(1)

    print("=" * 60)
    print(f"  {config['description']}")
    print(f"  Période : {config.get('date_debut') or 'début'} -> {config.get('date_fin') or 'fin'}")
    print("=" * 60)

    metadata = fetch_station_metadata()
    df_all = download_all(config, metadata)

    output_file = os.path.join("data", config["output_file"])
    df_all.to_csv(output_file, sep=";", index=False, encoding="utf-8-sig", float_format="%.2f")

    print(f"\n{'=' * 60}")
    print(f"  RÉSULTAT : {len(df_all)} lignes {df_all['station_name'].nunique()} stations")
    print(f"  Colonnes : {list(df_all.columns)}")
    print(f"  Fichier  : {output_file}")
    print(f"{'=' * 60}")
    print("\nAperçu :")
    print(df_all.head(6).to_string())
