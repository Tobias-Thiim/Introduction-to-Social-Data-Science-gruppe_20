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
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

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








def prepare_ml_dataset(real_estate_with_density):
    ml_dataset = real_estate_with_density[['priceCash', 'job_density', 'distance_to_job_center', 'job_count_5km']]
    return ml_dataset.dropna()

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
    
    # Add title and labels
    plt.title("Geographic Heatmap of House Prices with Job Centers")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.legend()
    
    plt.show()

def plot_house_prices_by_job_density(ml_dataset):
    ml_dataset['job_density_quartile'] = pd.qcut(ml_dataset['job_density'], q=4, labels=['Low', 'Medium-Low', 'Medium-High', 'High'])

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
