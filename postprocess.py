# Math imports
import shutil
import time

import numpy as np
import xarray as xr
import pandas as pd

# GIS imports
import rasterio.crs
import rasterio
import rioxarray
from rasterio.rio.options import resolution_opt
from rioxarray import merge
import geopandas as gpd
from pyogrio import set_gdal_config_options
from shapely.geometry import mapping
from osgeo import gdal, ogr

# General imports
import os, glob, shutil
from multiprocessing import Pool
import matplotlib.pyplot as plt
import cartopy.io.img_tiles as cimgt

# Set GDAL configuration
set_gdal_config_options({'SHAPE_RESTORE_SHX': 'YES'})


def process_date(s_target_directory: str, s_date: str, s_shapefile: object, s_output_directory: str, s_output_name: str, t_figure_size: tuple):
    """
    Processes the GFS based ET forecasts for a specific date across all calculated tiles. It returns a pandas dataframe that will be combined with other dates

    Parameters
    ----------
    s_target_directory: str
        Path to the GFS output
    s_date: str
        Date in ET Toolbox foramt
    s_shapefile: str
        Path to the shapefile for the area being summarized
    s_output_directory: str
        Path to copy the output rater files into
    s_output_name: str
        Name of the output file
    t_figure_size: tuple
        Size of the map to be created

    Returns
    -------
    df_output: pd.DataFrame
        Contains the output for the daily summary

    """


    # Create a dataframe to hold the data
    df_output = pd.DataFrame(index=[s_date], columns=['count', 'area', 'minimum', 'maximum', 'range', 'mean', 'std', 'percentile_90'])

    # Create a working path
    s_working_date_path = os.path.join(s_target_directory, s_date)

    # Search the folder structure using glob for ET tiles
    sl_working_date_et_tiles = glob.glob(os.path.join(s_working_date_path, '*', '*_ET.tif'), recursive=True)
    #sl_working_date_et_tiles = glob.glob(os.path.join(s_working_date_path, '*_ET.tif'), recursive=True)

    # Attempt to process the data
    try:
        # Clip the datasets
        s_raster_output_path = os.path.join(s_working_date_path, s_output_name + '_clipped.tiff')
        gdal.Warp(s_raster_output_path, sl_working_date_et_tiles, format="GTiff", resampleAlg='average', cutlineDSName=s_shapefile, cropToCutline=True,
                  creationOptions=["COMPRESS=LZW"])

        # Read the datasets
        dx_dataset = xr.open_dataset(s_raster_output_path, engine='rasterio')['band_data']

        # Perform math
        df_output.loc[s_date, 'count'] = np.sum(~np.isnan(dx_dataset)).values
        df_output.loc[s_date, 'area'] = df_output.loc[s_date, 'count'] * 30 * 30
        df_output.loc[s_date, 'minimum'] = np.nanmin(dx_dataset)
        df_output.loc[s_date, 'maximum'] = np.nanmax(dx_dataset)
        df_output.loc[s_date, 'range'] = df_output.loc[s_date]['maximum'] - df_output.loc[s_date]['minimum']
        df_output.loc[s_date, 'mean'] = np.nanmean(dx_dataset)
        df_output.loc[s_date, 'std'] = np.nanstd(dx_dataset)
        df_output.loc[s_date, 'percentile_90'] = np.percentile(dx_dataset.values.flatten()[~np.isnan(dx_dataset.values.flatten())], 90)

    except:
        # Something went wrong in the data processing. Return the empty frame or the current calculations
        pass

    # Attempt to create the output folder
    try:
        # Create the output folder
        s_daily_output = os.path.join(s_output_directory, s_date)

        if not os.path.isdir(s_daily_output):
            os.makedirs(s_daily_output)

    except:
        # Issue with creating the folder. Skip creating it
        pass

    # Create the map of the tiles
    try:
        create_map(dx_dataset, s_daily_output, s_output_name, t_figure_size)

    except:
        pass

    # Attempt to move the rasters
    try:
        # Copy the ET files to the output directory
        for i_entry_file in range(0, len(sl_working_date_et_tiles), 1):
            # Create the new filename
            s_new_path = os.path.join(s_daily_output, os.path.basename(sl_working_date_et_tiles[i_entry_file]))

            # Copy the file
            if not os.path.isfile(s_new_path):
                shutil.copyfile(sl_working_date_et_tiles[i_entry_file], s_new_path)

    except:
        pass

    # Return the dataframe
    return df_output

def create_map(o_merged_datasets: xr.DataArray, s_output_directory: str, s_output_name: str, t_figure_size: tuple):
    """
    Creates a map of an output region with the raster over the top of a basemap

    Parameters
    ----------
    o_merged_datasets: xr.DataArray
        Contains the clipped output data
    s_output_directory: str
        Path to the output directory
    s_output_name: str
        Name of the output file
    t_figure_size: tuple
        Figure size

    Returns
    -------
    None. The map is written to disk

    """

    # Use OpenStreetMap tiles
    o_osm_tiles = cimgt.OSM()

    # Define projection
    s_projection = o_osm_tiles.crs
    dx_data_projected = o_merged_datasets.rio.reproject(s_projection)

    # Create the figure
    o_figure, o_axis = plt.subplots(figsize=t_figure_size, subplot_kw={'projection': s_projection})

    # Add basemap
    o_axis.add_image(o_osm_tiles, 10)

    # Add raster data
    o_mesh = o_axis.pcolormesh(dx_data_projected['x'], dx_data_projected['y'], dx_data_projected[0], cmap='Reds', alpha=0.6, shading='auto')

    # Add colorbar
    o_color_bar = plt.colorbar(o_mesh, ax=o_axis, orientation='vertical', pad=0.03)

    # Adjust layout
    plt.tight_layout()

    # Save and close the file
    plt.savefig(os.path.join(s_output_directory, s_output_name + '.png'), dpi=600)
    plt.close()


def process_shapefile(s_input_directory, s_target_shapefile: str, s_output_directory: str, s_output_name: str, sl_dates: list, t_figure_size: tuple):
    """
    Process each of the shapefiles to create a summary file and a raster map

    Parameters
    ----------
    s_input_directory: str
        Path to the ET Toolbox directory containing results
    s_target_shapefile: str
        Path to the target shapefile
    s_output_directory: str
        Path to the output folder
    s_output_name: str
        Specified output name
    sl_dates: list
        List of dates to analyze within the current window
    t_figure_size: tuple
        Size of the map to be created

    Returns
    -------
    None. Data is written to disk

    """

    ### Create the pandas dataframe to hold the output ###
    df_output = pd.DataFrame(index=sl_dates, columns=['count', 'area', 'minimum', 'maximum', 'range', 'mean', 'std', 'percentile_90'])

    ### Loop and process each date ###
    # Serial processing
    #ol_tasks = [process_date(s_input_directory, x, s_target_shapefile, s_output_directory, s_output_name, t_figure_size) for x in sl_dates]

    # Parallel processing
    # Open the compute pool
    o_pool = Pool()

    # Create the tasks
    ol_tasks = [o_pool.apply_async(process_date, args=(s_input_directory, x, s_target_shapefile, s_output_directory, s_output_name, t_figure_size)) for x in sl_dates]

    # Get the tasks
    ol_tasks = [x.get() for x in ol_tasks]

    # Close the compute pool
    o_pool.close()

    ### Post process the output ###
    try:
        # Concatenate the output
        df_output = pd.concat(ol_tasks, axis=0)

        # Save the dataframe to csv for later use
        s_filename = 'summary_' + s_output_name + '.csv'
        df_output.to_csv(s_filename)

        # Copy the summary file to the drive
        shutil.copyfile(s_filename, os.path.join(s_output_directory, s_filename))

    except:
        # Error with the output. Just pass and not create the file
        pass

def remove_directory(s_output_directory):
    """
    Removes target directory to allow for new data

    Parameters
    ----------
    s_output_directory: str
        Path to target directory.

    Returns
    -------
    None. File is written to disk.

    """

    # Make sure the output directory exists
    if not os.path.isdir(s_output_directory):
        os.makedirs(s_output_directory)

    else:
        # Remove the existing folder
        shutil.rmtree(s_output_directory)

        # Pause to allow the delete
        time.sleep(15)

        # Remake the directory
        os.makedirs(s_output_directory)

if __name__ == '__main__':

    ### Process the forecast outputs ###
    # Set the input directories
    s_input_directory = 'GFS_output'
    sl_target_shapefiles = [os.path.join("shapefiles", 'mrg', "Projects.shp"),
                            os.path.join("shapefiles", 'riparian', "RG_Riparian_2km.shp")]
    tl_figure_sizes = [(3.25, 8), (3.25, 8)]

    # Set the output directory
    s_output_directory = '/mnt/export/et_rasters/forecast'
    sl_output_names = ['forecast_mrg', 'forecast_riparian']

    # Remove the output directory
    remove_directory(s_output_directory)

    # Process if the folder exists
    if os.path.isdir(s_input_directory):
        # Find the dates in the folder
        sl_dates = os.listdir(s_input_directory)

        # Process each of the shapefiles
        for i_entry_shapefile in range(0, len(sl_target_shapefiles), 1):
            process_shapefile(s_input_directory, sl_target_shapefiles[i_entry_shapefile], s_output_directory, sl_output_names[i_entry_shapefile], sl_dates,
                              tl_figure_sizes[i_entry_shapefile])

    ### Process the hindcast outputs ###
    # Set the input directories
    s_input_directory = 'LANCE_output'

    # Set the output directory
    s_output_directory = '/mnt/export/et_rasters/hindcast'
    sl_output_names = ['hindcast_mrg', 'hindcast_riparian']

    # Remove the output directory
    remove_directory(s_output_directory)

    # Process if the folder exists
    if os.path.isdir(s_input_directory):
        # Find the dates in the folder
        sl_dates = os.listdir(s_input_directory)

        # Process each of the shapefiles
        if os.path.isdir(s_input_directory):
            for i_entry_shapefile in range(0, len(sl_target_shapefiles), 1):
                process_shapefile(s_input_directory, sl_target_shapefiles[i_entry_shapefile], s_output_directory, sl_output_names[i_entry_shapefile], sl_dates,
                                  tl_figure_sizes[i_entry_shapefile])
