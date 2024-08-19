import requests
import tqdm
import time
import pandas as pd
import json
import os
import numpy as mp
import seaborn as sns
import geopandas as gpd
from shapely.geometry import Point, box
import numpy as np
from scipy.spatial import cKDTree
import ast
import matplotlib.pyplot as plt
from pyproj import Transformer
from geopy.distance import geodesic
from scipy.spatial import KDTree
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


# Class for scraping Boliga data
class BoligaScraper:
    def __init__(self, completed_pages_file='completed_boliga_pages.json', logfile='scrape_log.txt'):
        self.completed_pages_file = completed_pages_file
        self.completed_pages = self.load_completed_pages()
        self.logfile = logfile
        self.json_list = []

    def load_completed_pages(self):
        if os.path.exists(self.completed_pages_file):
            try:
                with open(self.completed_pages_file, 'r') as f:
                    return set(json.load(f))
            except json.JSONDecodeError:
                print(f"Warning: Could not decode JSON from {self.completed_pages_file}. Initializing an empty set.")
        return set()

    def save_completed_pages(self):
        with open(self.completed_pages_file, 'w') as f:
            json.dump(list(self.completed_pages), f)

    def scrape(self):
        links = []
        for page in range(1, 1104):
            if page in self.completed_pages:
                print(f"Skipping already downloaded page {page}")
                continue
            url = f"https://api.boliga.dk/api/v2/search/results?sort=views-d&page={page}"
            links.append((url, page))

        for url, page in tqdm.tqdm(links):
            try:
                response = requests.get(url, headers={'name': 'Jesper B Petersen', 'email': 'kwf929@alumni.ku.dk'}, verify=False)
                response.raise_for_status()
            except Exception as e:
                print(url)
                print(e)
                continue

            json_data = response.json()
            self.json_list.append(json_data)
            self.completed_pages.add(page)
            self.save_completed_pages()
            time.sleep(2)
            log(response, self.logfile)

    def save_to_csv(self, output_file='boliger_salg_boliga.csv'):
        bolig_df = pd.DataFrame()
        for json_data in self.json_list:
            df = pd.DataFrame(json_data.get('results', []))
            bolig_df = pd.concat([bolig_df, df], ignore_index=True)

        pd.set_option('display.max_columns', None)
        bolig_df.to_csv(output_file, sep=';', index=False, encoding='utf-8')
        print(f"All data saved to {output_file}")


# Class for scraping boligsiden data
class RealEstateScraper:
    def __init__(self, completed_pages_file='completed_pages.json'):
        self.property_types = [
            "villa", "condo", "terraced house", "holiday house", "cooperative",
            "farm", "hobby farm", "full year plot", "villa apartment", 
            "holiday plot", "houseboat"
        ]
        self.completed_pages_file = completed_pages_file
        self.completed_pages = self.load_completed_pages()
        self.all_dataframes = []

    def load_completed_pages(self):
        if os.path.exists(self.completed_pages_file):
            try:
                with open(self.completed_pages_file, 'r') as f:
                    completed_pages = json.load(f)
                    for key in completed_pages:
                        completed_pages[key] = set(completed_pages[key])
                return completed_pages
            except json.JSONDecodeError:
                print(f"Warning: Could not decode JSON from {self.completed_pages_file}. Initializing a new empty dictionary.")
                return {}
        return {}

    def scrape(self):
        for property_type in self.property_types:
            print(f"Scraping data for: {property_type}")
            property_links = []
            for page in range(1, 201):
                if self.completed_pages.get(property_type, set()).issuperset({page}):
                    print(f"Skipping already downloaded page {page} for {property_type}")
                    continue
                
                url = f'https://api.boligsiden.dk/search/cases?addressTypes={property_type.replace(" ", "%20")}&per_page=50&page={page}&sortAscending=true&sortBy=timeOnMarket'
                property_links.append((url, page))

            all_real_estate_postings = []
            for url, page in tqdm.tqdm(property_links):
                try:
                    response = requests.get(url, headers={'name': 'Jesper Petersen, Københavns Universitet, Studerende, email:kwf929@alumni.ku.dk'})
                    response.raise_for_status()
                except Exception as e:
                    print(url)
                    print(e)
                    continue

                result = response.json()
                real_estate_postings = result.get('cases', [])
                if real_estate_postings is None:
                    print(f"Warning: Received 'None' for postings at page {page} for {property_type}")
                    continue
                
                all_real_estate_postings.extend(real_estate_postings)
                self.completed_pages.setdefault(property_type, set()).add(page)
                self.save_completed_pages()

                time.sleep(0.5)

            df = pd.DataFrame(all_real_estate_postings)
            self.all_dataframes.append(df)
            csv_filename = f'{property_type.replace(" ", "_")}_boligsiden_postings.csv'
            df.to_csv(csv_filename, index=False)
            print(f"Data for {property_type} saved to {csv_filename}")

        final_df = pd.concat(self.all_dataframes, ignore_index=True)
        final_csv_filename = 'final_boligsiden_postings.csv'
        final_df.to_csv(final_csv_filename, index=False)
        print(f"All data saved to {final_csv_filename}")

    def save_completed_pages(self):
        with open(self.completed_pages_file, 'w') as f:
            completed_pages_to_save = {k: list(v) for k, v in self.completed_pages.items()}
            json.dump(completed_pages_to_save, f)

# Class for scraping jobnet.dk data
class JobScraper:
    def __init__(self, completed_pages_file='completed_job_pages.json'):
        self.completed_pages_file = completed_pages_file
        self.completed_pages = self.load_completed_pages()
        self.all_job_postings = []

    def load_completed_pages(self):
        if os.path.exists(self.completed_pages_file):
            try:
                with open(self.completed_pages_file, 'r') as f:
                    return set(json.load(f))
            except json.JSONDecodeError:
                print(f"Warning: Could not decode JSON from {self.completed_pages_file}. Initializing an empty set.")
        return set()

    def scrape(self):
        links = []
        for page in range(0, 17992, 20):
            if page in self.completed_pages:
                print(f"Skipping already downloaded page {page}")
                continue
            url = f'https://job.jobnet.dk/CV/FindWork/Search?Offset={page}'
            links.append(url)

        logfile = 'log3.csv'
        for url in tqdm.tqdm(links):
            try:
                response = requests.get(url, headers={'name': 'Jesper Petersen, Københavns Universitet, Studerende, email:kwf929@alumni.ku.dk'})
                response.raise_for_status()
            except Exception as e:
                print(url)
                print(e)
                continue

            result = response.json()
            job_position_postings = result.get('JobPositionPostings', [])
            self.all_job_postings.extend(job_position_postings)
            self.completed_pages.add(page)
            self.save_completed_pages()
            time.sleep(0.5)
            log(response, logfile)

            Job_Position_Postings = pd.DataFrame(self.all_job_postings)
            Job_Position_Postings.to_csv("job_postings_backup.csv", index=False)

        final_csv_filename = 'final_jobnet_postings.csv'
        Job_Position_Postings.to_csv(final_csv_filename, index=False)
        print(f"All data saved to {final_csv_filename}")

    def save_completed_pages(self):
        with open(self.completed_pages_file, 'w') as f:
            json.dump(list(self.completed_pages), f)


def get_and_clean_stations():
    url_station = "https://www.dsb.dk/api/stations/getstationlist"
    
    # Step 1: Fetch station data
    response_station = requests.get(url_station)
    response_station_json = response_station.json()
    
    # Step 2: Convert response to DataFrame
    station_df = pd.DataFrame(response_station_json)
    
    # Step 3: Clean the latitude and longitude columns
    station_df['stationLatitude'] = station_df['stationLatitude'].str.replace(',', '.').astype(float)
    station_df['stationLongitude'] = station_df['stationLongitude'].str.replace(',', '.').astype(float)
    
    # Step 4: Clean station names
    def clean_station_name(name):
        return name.replace(' Station', '')
    
    station_df['stationName'] = [clean_station_name(name) for name in station_df['stationName']]
    
    # Step 5: Apply specific name replacements
    station_df['stationName'] = [
        'Viby Jylland' if station == 'Viby J' else
        'Nivå' if station == 'Nivå Station' else
        'Højby (Fyn)' if station == 'Højby Fyn station' else
        'CPH Lufthavn ✈︎' if station == 'Københavns Lufthavn (CPH Airport)' else
        'Nykbing F' if station == 'Nykøbing sj' else
        'Viby Sj' if station == 'Viby Sjælland' else
        'Aalborg Lufthavn ✈︎' if station == 'Aalborg Lufthavn' else
        station 
        for station in station_df['stationName']
    ]
    
    return station_df

# Define the log function to gather the log information
def log(response,logfile,output_path=os.getcwd()):
    if os.path.isfile(logfile):
        log = open(logfile,'a')
    else:
        log = open(logfile,'w')
        header = ['timestamp','status_code','length','output_file']
        log.write(';'.join(header) + "\n")
    
    status_code = response.status_code
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    length = len(response.text)
    
    with open(logfile,'a') as log:
        log.write(f'{timestamp};{status_code};{length};{output_path}' + "\n")

def parse_coordinates(coord):
    if isinstance(coord, str):
        try:
            return ast.literal_eval(coord)
        except (ValueError, SyntaxError):
            return None
    return None

def create_gdf_real_estate(df):
    df['coordinates'] = df['coordinates'].apply(parse_coordinates)
    df = df[df['coordinates'].notnull()].copy()
    df['lat'] = df['coordinates'].apply(lambda x: x['lat'])
    df['lon'] = df['coordinates'].apply(lambda x: x['lon'])
    geometry = [Point(xy) for xy in zip(df['lon'], df['lat'])]
    return gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")

def create_gdf_real_estate_boliga(df):
    # Ensure no missing values
    df = df[df['latitude'].notnull() & df['longitude'].notnull()].copy()
    
    # Create geometry using longitude and latitude
    geometry = [Point(xy) for xy in zip(df['longitude'], df['latitude'])]
    
    # Create GeoDataFrame
    return gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")


def create_gdf_jobs(df):
    geometry = [Point(xy) for xy in zip(df['Longitude'], df['Latitude'])]
    return gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")

def create_grid(real_estate_gdf, cell_size=1000):
    real_estate_gdf_proj = real_estate_gdf.to_crs(epsg=25832)
    bounds = real_estate_gdf_proj.total_bounds
    grid_cells_proj = gpd.GeoDataFrame(
        geometry=[
            box(x, y, x + cell_size, y + cell_size)
            for x in np.arange(bounds[0], bounds[2], cell_size)
            for y in np.arange(bounds[1], bounds[3], cell_size)
        ],
        crs="EPSG:25832"
    )
    return grid_cells_proj.to_crs(epsg=4326)

def calculate_job_density(grid_cells, job_listings_gdf):
    job_counts = gpd.sjoin(job_listings_gdf, grid_cells, how="inner", predicate="within")
    job_density = job_counts.groupby(job_counts.index_right).size().rename('job_density')
    if 'job_density' in grid_cells.columns:
        grid_cells = grid_cells.drop(columns=['job_density'])
    grid_cells = grid_cells.join(job_density, how='left')
    grid_cells['job_density'] = grid_cells['job_density'].fillna(0)
    return grid_cells




def nearest_neighbor(point_gdf, centers_gdf, k=1):
    # Ensure we're working with centroids for both datasets
    point_centroids = point_gdf.geometry.centroid
    center_centroids = centers_gdf.geometry.centroid
    
    # Project the geometries to a planar coordinate system (e.g., UTM)
    point_gdf_proj = point_centroids.to_crs(epsg=3395)  # WGS 84 / World Mercator
    centers_proj = center_centroids.to_crs(epsg=3395)
    
    # Extract the projected coordinates
    points = np.array([(p.x, p.y) for p in point_gdf_proj])
    centers = np.array([(c.x, c.y) for c in centers_proj])
    
    # Create cKDTree for the centers
    btree = cKDTree(centers)
    
    # Get the nearest neighbors and distances
    distances, indices = btree.query(points, k=k)
    
    # Convert distances back to kilometers using the geodesic distance for accuracy
    km_distances = []
    for i, point in enumerate(point_centroids):
        nearest_center = center_centroids.iloc[indices[i]]
        km_distance = geodesic((point.y, point.x), (nearest_center.y, nearest_center.x)).kilometers
        km_distances.append(km_distance)
    
    return np.array(km_distances)



def calculate_job_count_5km(real_estate_gdf, job_listings_gdf, projected_crs="EPSG:25832"):
    """
    Calculate the job count within 5km for each real estate entry.
    
    Parameters:
    - real_estate_gdf: Existing GeoDataFrame containing real estate data.
    - job_listings_gdf: GeoDataFrame containing job listings data.
    - projected_crs: The projected CRS to use for calculations (default is EPSG:25832).
    
    Returns:
    - job_count_5km: Series with job counts, indexed to match the real_estate_gdf index.
    """
    # Step 1: Reproject GeoDataFrames to a projected CRS
    real_estate_gdf_proj = real_estate_gdf.to_crs(projected_crs)
    job_listings_gdf_proj = job_listings_gdf.to_crs(projected_crs)
    
    # Step 2: Perform the nearest spatial join with the projected CRS
    joined = gpd.sjoin_nearest(
        real_estate_gdf_proj, 
        job_listings_gdf_proj, 
        max_distance=5000,
        how='left'  # Ensure we keep all real estate entries even if no job is nearby
    )
    
    # Step 3: Calculate job count per property
    job_count_5km = joined.groupby(joined.index).size()

    # Step 4: Ensure the result is a Series and has the correct index
    job_count_5km_full = pd.Series(0, index=real_estate_gdf.index)
    job_count_5km_full.update(job_count_5km)

    return job_count_5km_full



def process_and_calculate_nearest_station(real_estate_df, station_df_med_afgange):
    # Function to safely convert to float
    def safe_float_convert(value):
        if isinstance(value, (float, int)):
            return float(value)
        elif isinstance(value, str):
            return float(value.replace(',', '.'))
        else:
            raise ValueError(f"Unexpected type: {type(value)}")
    
    # Apply the conversion
    station_df_med_afgange['stationLatitude'] = station_df_med_afgange['stationLatitude'].apply(safe_float_convert)
    station_df_med_afgange['stationLongitude'] = station_df_med_afgange['stationLongitude'].apply(safe_float_convert)
    
    # Extract coordinates
    house_coords = real_estate_df[['latitude', 'longitude']].values
    station_coords = station_df_med_afgange[['stationLatitude', 'stationLongitude']].values
    
    # Create KDTree
    station_tree = KDTree(station_coords)
    
    # Function to find nearest station info
    def find_nearest_station_info(house_coord, station_tree, stations, station_df):
        distance, index = station_tree.query(house_coord, k=1)
        nearest_station = stations[index]
        geodesic_distance = geodesic(house_coord, nearest_station).kilometers
        station_name = station_df.iloc[index]['Station_x']
        departures = station_df.iloc[index]['Afgange_x']
        return geodesic_distance, station_name, departures
    
    # Apply function to all houses
    nearest_station_info = [find_nearest_station_info(house, station_tree, station_coords, station_df_med_afgange) for house in house_coords]
    
    # Add new columns
    real_estate_df['distance_to_nearest_station'] = [info[0] for info in nearest_station_info]
    real_estate_df['nearest_station_name'] = [info[1] for info in nearest_station_info]
    real_estate_df['departures_per_hour'] = [info[2] for info in nearest_station_info]
    
    return real_estate_df



def prepare_ml_dataset(real_estate_with_density):
    # Step 1: Filter out the unwanted address types
    excluded_types = ["holiday house", "cooperative", "holiday plot", "houseboat"]
    filtered_df = real_estate_with_density[~real_estate_with_density['addressType'].isin(excluded_types)]

    # Step 2: Select the original and new variables for machine learning
    ml_dataset = filtered_df[[
        'priceCash', 'distance_to_nearest_station', 'job_density', 'distance_to_job_center', 'job_count_5km', 
        'distinction', 'daysOnMarket', 'energyLabel', 'hasBalcony', 'hasElevator', 
        'hasTerrace', 'highlighted', 'housingArea', 'lotArea', 'monthlyExpense',
        'numberOfBathrooms', 'numberOfFloors', 'numberOfRooms', 'numberOfToilets', 
        'pageViews', 'perAreaPrice', 'realtor', 'timeOnMarket', 'utilitiesConnectionFee', 
        'weightedArea', 'yearBuilt', 'basementArea', 'lat', 'lon', 'geometry'
    ]]
    
    # Step 3: Drop any rows with NaN values
    ml_dataset = ml_dataset.dropna()

    return ml_dataset

def prepare_ml_dataset_boliga(real_estate_with_density):
    # Step 1: Filter out the unwanted address types
    excluded_types = ["4", "5", "8"]
    filtered_df = real_estate_with_density[~real_estate_with_density['propertyType'].isin(excluded_types)]

    # Step 2: Rename 'price' to 'priceCash'
    filtered_df = filtered_df.rename(columns={'price': 'priceCash'})

    # Step 3: Filter out rows where latitude and longitude are outside the Denmark bounds
    filtered_df = filtered_df[
        (filtered_df['latitude'] >= 54.5) & (filtered_df['latitude'] <= 57.8) &
        (filtered_df['longitude'] >= 7.5) & (filtered_df['longitude'] <= 12.7)
    ]

    # Step 4: Select the original and new variables for machine learning
    ml_dataset = filtered_df[[
        'geometry', 'job_density', 'distance_to_nearest_station', 'distance_to_job_center', 'job_count_5km', 'latitude', 'longitude', 'propertyType', 'energyClass', 'priceCash', 'selfsale', 'rooms', 'size', 'lotSize', 'floor', 'buildYear', 'city', 'isForeclosure', 'zipCode', 'area', 'daysForSale', 'net', 'exp', 'basementSize', 'views', 'projectSaleUrl', 'additionalBuildings', 'businessArea', 'nonPremiumDiscrete', 'cleanStreet'
    ]]

    # Step 5: Drop rows only if there are missing values in the key columns
    ml_dataset = ml_dataset.dropna(subset=['latitude', 'longitude', 'priceCash', 'size'])

    return ml_dataset



def prepare_ml_dataset_overall(real_estate_df):
    # Step 0: Initial number of listings
    initial_count = len(real_estate_df)
    print(f"Step 0: Initial number of listings: {initial_count}")
    
    # Step 1: Filter out the unwanted address types (4: Andelsboliger, 5: Fritidshuse, 8: Fritidsgrunde)
    excluded_types = [4, 5, 8]
    filtered_df = real_estate_df[~real_estate_df['propertyType'].isin(excluded_types)].reset_index(drop=True)
    step1_count = len(filtered_df)
    print(f"Step 1: Filter out unwanted address types. Removed {initial_count - step1_count} listings. Remaining: {step1_count}")
    
    # Step 2: Rename 'price' to 'priceCash'
    filtered_df = filtered_df.rename(columns={'price': 'priceCash'})
    print(f"Step 2: Rename 'price' to 'priceCash'. Removes none. Remaining: {step1_count}")
    
    # Step 3: Filter out rows where latitude and longitude are outside the Denmark bounds
    before_step3_count = len(filtered_df)
    filtered_df = filtered_df[
        (filtered_df['latitude'] >= 54.5) & (filtered_df['latitude'] <= 57.8) &
        (filtered_df['longitude'] >= 7.5) & (filtered_df['longitude'] <= 15.3)
    ]
    step3_count = len(filtered_df)
    print(f"Step 3: Filter out long+lat outside of Denmark. Removed {before_step3_count - step3_count} listings. Remaining: {step3_count}")
    
    # Step 4: Filter out properties with price less than 29,500
    before_step4_count = len(filtered_df)
    filtered_df = filtered_df[filtered_df['priceCash'] > 29500]
    step4_count = len(filtered_df)
    print(f"Step 4: Filter out properties with price < 29,500. Removed {before_step4_count - step4_count} listings. Remaining: {step4_count}")

    # Step 5: Remove properties with size less than 10 sqm unless they are of specific property types (7 or 8)
    before_step5_count = len(filtered_df)
    filtered_df = filtered_df.loc[
        (filtered_df['size'] >= 10) | 
        (filtered_df['propertyType'].isin([7, 8]))
    ]
    step5_count = len(filtered_df)
    print(f"Step 5: Remove properties with size < 10 sqm. Removed {before_step5_count - step5_count} listings. Remaining: {step5_count}")
    
    # Step 6: Remove specific properties (garages) identified by street names
    before_step6_count = len(filtered_df)
    filtered_df = filtered_df.loc[
        (~filtered_df['street'].isin(["Løvholmen 14, st.. 10.", "Store Kongensgade 90, st.. 4."]))
    ]
    step6_count = len(filtered_df)
    print(f"Step 6: Remove specific properties (garages). Removed {before_step6_count - step6_count} listings. Remaining: {step6_count}")
    
    # Step 7: Sort by 'createdDate' in ascending order to ensure the oldest observations are kept
    filtered_df = filtered_df.sort_values(by='createdDate')
    print(f"Step 7: Sort by 'createdDate'. No removals. Remaining: {step6_count}")
    
    # Step 8: Split the dataframe based on PropertyType
    df_type1 = filtered_df[filtered_df['propertyType'] == 1]
    df_type2 = filtered_df[filtered_df['propertyType'] == 2]
    df_type3 = filtered_df[filtered_df['propertyType'] == 3]
    df_type6 = filtered_df[filtered_df['propertyType'] == 6]
    df_type7 = filtered_df[filtered_df['propertyType'] == 7]

    # Step 9: Drop duplicates based on certain columns depending on the property type
    before_step9_count = len(filtered_df)
    df_type1 = df_type1.drop_duplicates(subset=['street', 'zipCode'])
    df_type2 = df_type2.drop_duplicates(subset=['street', 'zipCode', 'size'])
    df_type3 = df_type3.drop_duplicates(subset=['street', 'zipCode', 'size'])
    df_type6 = df_type6.drop_duplicates(subset=['street', 'zipCode', 'lotSize'])
    df_type7 = df_type7.drop_duplicates(subset=['street', 'zipCode'])
    filtered_df = pd.concat([df_type1, df_type2, df_type3, df_type6, df_type7])
    step9_count = len(filtered_df)
    print(f"Step 9: Drop duplicates. Removed {before_step9_count - step9_count} listings. Remaining: {step9_count}")
    
    # Step 10: Reset index if needed
    filtered_df.reset_index(drop=True, inplace=True)
    print(f"Step 10: Reset index. No removals. Remaining: {step9_count}")
    
    # Step 11: Correct energyClass to uppercase and handle specific energyClass replacements
    filtered_df['energyClass'] = filtered_df['energyClass'].str.upper()
    mapping = {'M': 'A2', 'K': 'A10', 'J': 'A15', 'I': 'A20'}
    filtered_df['energyClass'] = filtered_df['energyClass'].replace(mapping)
    print(f"Step 11: Correct energyClass to uppercase and handle specific replacements. No removals. Remaining: {step9_count}")
    
    # Step 12: Set 'rooms' to NaN where 'rooms' == 0 and 'propertyType' is not 7 or 8
    filtered_df.loc[(filtered_df['rooms'] == 0) & (~filtered_df['propertyType'].isin([7, 8])), 'rooms'] = np.nan
    print(f"Step 12: Set 'rooms' to NaN where 'rooms' == 0 and 'propertyType' is not 7 or 8. No removals. Remaining: {step9_count}")
    
    # Step 13: Set 'lotSize' to NaN where it is 0 or 1 and the property is not an apartment (propertyType 3)
    filtered_df.loc[(filtered_df['lotSize'].isin([0, 1])) & (filtered_df['propertyType'] != 3), 'lotSize'] = np.nan
    print(f"Step 13: Set 'lotSize' to NaN where it is 0 or 1 and the property is not an apartment. No removals. Remaining: {step9_count}")
    
    # Step 14: Set 'buildYear' to NaN if it is before 1575 or if the property type is 7 or 8 (grunds or grunde)
    filtered_df.loc[(filtered_df['buildYear'] < 1575), 'buildYear'] = np.nan
    filtered_df.loc[(filtered_df['propertyType'].isin([7, 8])), 'buildYear'] = np.nan
    print(f"Step 14: Set 'buildYear' to NaN if it is before 1575 or if the property type is 7 or 8. No removals. Remaining: {step9_count}")

    # Step 15: Replace propertyType 11 and 12 with 10
    filtered_df['propertyType'] = filtered_df['propertyType'].replace([11, 12], 10)
    print(f"Step 15: Replace propertyType 11 and 12 with 10. No removals. Remaining: {step9_count}")
    
    # Step 16: Set municipality to NaN where it is 0 or -1
    filtered_df.loc[(filtered_df['municipality'].isin([0, -1])), 'municipality'] = np.nan
    print(f"Step 16: Set municipality to NaN where it is 0 or -1. No removals. Remaining: {step9_count}")
    
    # Step 17: Select the original and new variables for machine learning
    ml_dataset = filtered_df[[
        'geometry', 'job_density', 'distance_to_nearest_station', 'nearest_station_name', 'departures_per_hour', 'distance_to_job_center', 'job_count_5km', 
        'latitude', 'longitude', 'propertyType', 'energyClass', 'priceCash', 'selfsale', 'rooms', 'size', 
        'lotSize', 'floor', 'buildYear', 'city', 'isForeclosure', 'zipCode', 'area', 'daysForSale', 
        'net', 'exp', 'basementSize', 'views', 'projectSaleUrl', 'additionalBuildings', 'businessArea', 
        'nonPremiumDiscrete', 'cleanStreet'
    ]]
    final_count = len(ml_dataset)
    print(f"Step 17: Select variables for machine learning. Final number of listings: {final_count}")
    
    # Step 18: Drop rows only if there are missing values in the key columns
    ml_dataset = ml_dataset.dropna(subset=['latitude', 'longitude', 'priceCash', 'size'])
    print(f"Step 18: Drop rows with missing key values. Final number of listings: {len(ml_dataset)}")

    return ml_dataset




def plot_price_vs_job_density(ml_dataset):
    """Plot Property Price vs Job Density"""
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='job_density', y='priceCash', data=ml_dataset)
    plt.title("Property Price vs Job Density")
    plt.xlabel("Job Density")
    plt.ylabel("Price (DKK)")
    plt.show()

def plot_price_vs_distance_to_job_center(ml_dataset):
    """Plot Property Price vs Distance to Nearest Job Center"""
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='distance_to_job_center', y='priceCash', data=ml_dataset)
    plt.title("Property Price vs Distance to Nearest Job Center")
    plt.xlabel("Distance to Nearest Job Center (degrees)")
    plt.ylabel("Price (DKK)")
    plt.show()

def plot_price_by_job_count_5km(ml_dataset):
    """Plot Property Price by Job Count Within 5km"""
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='job_count_5km', y='priceCash', data=ml_dataset)
    plt.title("Property Price by Job Count Within 5km")
    plt.xlabel("Job Count Within 5km")
    plt.ylabel("Price (DKK)")
    plt.xticks(rotation=45)
    plt.show()

def plot_heatmap_house_prices(real_estate_with_density):
    """Plot Geographic Heatmap of House Prices"""
    plt.figure(figsize=(12, 10))
    real_estate_with_density.plot(
        column='priceCash', 
        cmap='YlOrRd', 
        scheme='quantiles', 
        legend=False,
        markersize=3,  
        alpha=0.7  
    )
    plt.title("Geographic Heatmap of House Prices")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.show()

def plot_heatmap_house_prices_boliga(ml_dataset):
    """Plot Geographic Heatmap of House Prices"""
    plt.figure(figsize=(12, 10))
    
    # Ensure the geometry is used for plotting
    ml_dataset.plot(
        column='priceCash', 
        cmap='YlOrRd', 
        scheme='quantiles', 
        legend=False,  # Show the legend to understand the color scale
        markersize=3,  # Increase marker size for better visibility if needed
        alpha=0.7  
    )
    
    plt.title("Geographic Heatmap of House Prices")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.show()

def plot_heatmap_with_jobs_overlaid(real_estate_with_density, job_centers):
    """Plot Geographic Heatmap of House Prices with Jobs"""
    # Create the base map with the geographic heatmap of house prices
    plt.figure(figsize=(12, 10))
    
    base = real_estate_with_density.plot(
        column='priceCash', 
        cmap='YlOrRd', 
        scheme='quantiles', 
        markersize=2,  # Smaller marker size for house prices
        legend=False,
        alpha=0.5  # Transparency for house prices
    )
    
    # Overlay the job centers on top
    job_centers.plot(
        ax=base, 
        color='blue', 
        edgecolor='blue', 
        markersize=4,  # Slightly larger marker size for job centers
        alpha=0.4,  # Transparency for job centers
        marker='o',  # Circle markers for job centers
        label='Job Centers'
    )
    
    # Add a legend for job centers
    handles, labels = base.get_legend_handles_labels()
    if handles and labels:
        plt.legend(handles=handles, labels=labels)
    
    plt.title("Geographic Heatmap of House Prices with Job Centers")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    
    plt.show()

def plot_heatmap_with_jobs_overlaid_boliga(ml_dataset, job_centers):
    """Plot Geographic Heatmap of House Prices with Jobs"""
    # Create the base map with the geographic heatmap of house prices
    plt.figure(figsize=(12, 10))
    
    base = ml_dataset.plot(
        column='priceCash', 
        cmap='YlOrRd', 
        scheme='quantiles', 
        markersize=2,  # Smaller marker size for house prices
        legend=False,
        alpha=0.5  # Transparency for house prices
    )
    
    # Overlay the job centers on top
    job_centers.plot(
        ax=base, 
        color='blue', 
        edgecolor='blue', 
        markersize=4,  # Slightly larger marker size for job centers
        alpha=0.4,  # Transparency for job centers
        marker='o',  # Circle markers for job centers
        label='Job Centers'
    )
    
    # Add a legend for job centers
    handles, labels = base.get_legend_handles_labels()
    if handles and labels:
        plt.legend(handles=handles, labels=labels)
    
    plt.title("Geographic Heatmap of House Prices with Job Centers")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    
    plt.show()

def plot_house_prices_by_job_density(ml_dataset):
    # Define custom bins for job density
    bins = [0, 1, 10, 50, ml_dataset['job_density'].max()]
    labels = ['Very Low', 'Low', 'Medium', 'High']

    ml_dataset['job_density_quartile'] = pd.cut(ml_dataset['job_density'], bins=bins, labels=labels)

    plt.figure(figsize=(12, 6))
    sns.boxplot(x='job_density_quartile', y='priceCash', data=ml_dataset)
    plt.title("House Prices by Job Density Quartiles")
    plt.xlabel("Job Density Quartile")
    plt.ylabel("Price (DKK)")
    plt.show()




def dist_distance_nearest_center(ml_dataset):
    plt.figure(figsize=(10, 6))
    sns.histplot(data=ml_dataset, x='distance_to_job_center', kde=True)
    plt.title("Distribution of Distances to Nearest Job Center")
    plt.xlabel("Distance to Nearest Job Center (kilometers)")
    plt.show()

def regression_plot_houseprices_jobcount(ml_dataset):
    plt.figure(figsize=(12, 6))
    sns.regplot(x='job_count_5km', y='priceCash', data=ml_dataset, scatter_kws={'alpha':0.5})
    plt.title("Regression Plot: House Price vs Job Count within 5km")
    plt.xlabel("Number of Jobs within 5km")
    plt.ylabel("Price (DKK)")
    plt.show()
