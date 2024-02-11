import requests
import pandas as pd
import numpy as np
from typing import Tuple, List

from config import DATA
from utils import eyecolor_mapping

def get_phenotype_data() -> pd.DataFrame:
    """
    Fetches phenotype data from 'http://opensnp.org/phenotypes.json' and returns it as a pandas DataFrame.

    Returns:
        pd.DataFrame: The fetched phenotype data as a pandas DataFrame.
    """
    print("Fetching phenotype data...")
    url = 'http://opensnp.org/phenotypes.json'
    response = requests.get(url)
    if response.status_code == 200:
        df_pheno = pd.DataFrame(response.json())
        df_pheno.sort_values('number_of_users', ascending=False, inplace=True)
        df_pheno["number_variations"] = [len(known_variations) for known_variations in df_pheno["known_variations"]]
    else:
        print(f"Error fetching data: {response.status_code}")

    df_pheno.to_csv(DATA['PHEN'], index=False)
    print(df_pheno.head())
    return df_pheno

def get_user_data() -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Fetches user data from http://opensnp.org/phenotypes/json/variations/1.json and processes it.

    Returns:
        df_users (pd.DataFrame): DataFrame containing user data.
        user_ids (np.ndarray): Array of user IDs.
    """
    print("Fetching user data...")
    url = 'http://opensnp.org/phenotypes/json/variations/1.json'
    response = requests.get(url)
    if response.status_code == 200:
        df_users = pd.DataFrame(response.json()["users"])
    else:
        print(f"Error fetching data: {response.status_code}")

    df_users.drop_duplicates(inplace=True)

    df_users = eyecolor_mapping(df_users, "variation")

    print(f"Number of users with known eye-color: {df_users.shape[0]}")
    print(f"Eye-colors in the dataset: {df_users['variation'].unique()}")

    user_ids = df_users["user_id"].values

    df_users.to_csv(DATA['USER'], index=False)

    return df_users, user_ids

def get_annotations_data(rsids: List[str]) -> pd.DataFrame:
    """
    Fetches SNP annotations from opensnp.org and saves them to a CSV file.

    Args:
        rsids (List[str]): List of rsids to fetch annotations for.

    Returns:
        pd.DataFrame: DataFrame containing the fetched annotations.
    """
    print("Fetching rsn annotations...")

    rsids = ",".join(rsids)

    url = f"http://opensnp.org/snps/json/annotation/{rsids}.json"
    response = requests.get(url)
    if response.status_code == 200:
        annotations = response.json()
        df_annotations = pd.DataFrame(annotations)
    else:
        print(f"Error fetching annotations: {response.status_code}")
        df_annotations = pd.DataFrame()

    print(df_annotations)

    df_annotations.to_csv(DATA['ANNOTATIONS'], index=False)

    return df_annotations

def get_genotype_data(rsids: List[str], user_ids: List[int]) -> pd.DataFrame:
    """
    Fetches genotype data for the given rsids and user ids from an API and returns a DataFrame.

    Args:
        rsids (List[str]): List of rsids to fetch genotype data for.
        user_ids (List[int]): List of user ids to filter the genotype data.

    Returns:
        pd.DataFrame: DataFrame containing the fetched genotype data.

    Raises:
        None
    """
    print("Fetching genotype data...")

    # Define the base URL for the API
    base_url = "http://opensnp.org/snps/{}.json"
    
    # Initialize a dictionary to store the extracted information
    users_data = {}

    total_rsids = len(rsids)
    for idx, rsid in enumerate(rsids, start=1):
        print(f"Processing {idx} of {total_rsids}: {rsid}...")
        
        response = requests.get(base_url.format(rsid))

        if response.status_code == 200:
            data = response.json()
            for record in data:
                user_id = record['user']['id']
                
                if user_id in user_ids:
                    user_name = record['user']['name']
                    user_key = (user_name, user_id)

                    if record['user']['genotypes']:
                        local_genotype = record['user']['genotypes'][0]['local_genotype']
                        if local_genotype in ['--', '00']:
                            continue
                    else:
                        continue

                    if user_key not in users_data:
                        users_data[user_key] = {}

                    users_data[user_key][rsid] = local_genotype
        else:
            print(f"Error {response.status_code} fetching data for rsid: {rsid}")

    # Convert the nested dictionary into a DataFrame
    df = pd.DataFrame.from_dict(users_data, orient='index')
    df.reset_index(inplace=True)
    df.columns = ['user_name', 'user_id'] + rsids
    df.replace('--', 'NaN', inplace=True)

    print(f"Finished fetching genotype data")

    df.to_csv(DATA['GEN'], index=False)

    return df

def merge_users_genotypes(df_gen: pd.DataFrame, df_user: pd.DataFrame) -> pd.DataFrame:
    """
    Merge genotype and user data.

    Args:
        df_gen (pd.DataFrame): DataFrame containing genotype data.
        df_user (pd.DataFrame): DataFrame containing user data.

    Returns:
        pd.DataFrame: Merged DataFrame.

    """
    print("Merging genotype and user data...")
    df_gen = df_gen.copy()
    df_user = df_user.copy()

    df = df_gen.merge(df_user, on='user_id', how='left')

    df.rename(columns={'variation': 'eye_color'}, inplace=True)

    print(f"Final dataset has {df.shape[0]} users")

    df.to_csv(DATA['CLEAN'], index=False)

    return df


if __name__ == "__main__":
    # based on https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3694299/
    rsids = [
    "rs12896399", "rs12913832", "rs1545397", "rs16891982",
    "rs1426654", "rs885479", "rs6119471", "rs12203592"
    ] # to be validated

    df_pheno = get_phenotype_data()
    df_user, user_ids = get_user_data()
    df_annotations = get_annotations_data(rsids)
    df_gen = get_genotype_data(rsids, user_ids)
    df_clean = merge_users_genotypes(df_gen, df_user)
