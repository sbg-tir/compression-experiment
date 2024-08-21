import rasterio
from rasterio.enums import Compression
import numpy as np
import os
from os import makedirs
from os.path import join
import time
import pandas as pd

# Step 1: Load the input raster from 'input.tif' and ensure dtype is float32
input_file = 'input.tif'
output_directory = "output"
csv_filename = "results.csv"

makedirs(output_directory, exist_ok=True)

with rasterio.open(input_file) as src:
    data = src.read(1).astype(np.float32)  # Read the first band and convert to float32
    transform = src.transform
    crs = src.crs
    meta = src.meta

# Update metadata to ensure the dtype is float32
meta.update(dtype=rasterio.float32)

# Step 2: Save the raster as a COG with Deflate compression
uncompressed_filename = 'output_cog_uncompressed.tif'
uncompressed_meta = meta.copy()
uncompressed_meta.update({
    'driver': 'GTiff',
    'compress': None,
    'tiled': True,
    'blockxsize': 256,
    'blockysize': 256,
    'interleave': 'band'
})

with rasterio.open(uncompressed_filename, 'w', **uncompressed_meta) as dst:
    dst.write(data, 1)
    dst.build_overviews([2, 4, 8, 16])
    dst.update_tags(ns='rio_overview', resampling='nearest')

columns = [
    "precision_decimals",
    "zstd_level",
    "compressed_size_kb",
    "compression_percent",
    "write_time_milliseconds",
    "read_time_milliseconds",
    "RMSE"
]

df = pd.DataFrame([], columns=columns)

for precision_decimals in range(6):
    for zstd_level in range(22):
        print(f"testing compression with LERC {precision_decimals}-decimal precision and ZSTD level {zstd_level}")

        # Step 3: Save the raster as a COG with LERC compression and parameterize MAX_Z_ERROR
        max_z_error = 10 ** -precision_decimals  # Adjust this value to control compression accuracy

        lerc_filename = join(output_directory, f"output_cog_lerc_{precision_decimals}_zstd_{zstd_level}.tif")
        print(f"output file: {lerc_filename}")

        lerc_meta = meta.copy()
        lerc_meta.update({
            'driver': 'GTiff',
            'compress': "lerc_zstd",
            'tiled': True,
            'blockxsize': 256,
            'blockysize': 256,
            'interleave': 'band',
            "MAX_Z_ERROR": max_z_error,
            "zstd_level": zstd_level
        })

        start_time = time.perf_counter()

        with rasterio.open(lerc_filename, 'w', **lerc_meta) as dst:
            dst.write(data, 1)
            dst.build_overviews([2, 4, 8, 16])
            dst.update_tags(ns='rio_overview', resampling='nearest')
            dst.update_tags(LERC_MAXZERROR=max_z_error)

        end_time = time.perf_counter()
        write_time_milliseconds = int((end_time - start_time) * 1000)

        print(f"write time: {write_time_milliseconds} milliseconds")

        # Step 4: Compare the file sizes
        uncompressed_size_kb = os.path.getsize(uncompressed_filename) / 1024
        lerc_size_kb = os.path.getsize(lerc_filename) / 1024
        compression_percent = (uncompressed_size_kb - lerc_size_kb) / uncompressed_size_kb * 100

        print(f"Uncompressed file size: {uncompressed_size_kb:.2f} KB")
        print(f"File size with LERC compression: {lerc_size_kb:.2f} KB")
        print(f"Compression: {compression_percent:.2f}%")

        # Step 5: Measure the error between the Deflate and LERC data
        with rasterio.open(lerc_filename) as lerc_src:
            start_time = time.perf_counter()
            lerc_data = lerc_src.read(1).astype(np.float32)
            end_time = time.perf_counter()
            read_time_milliseconds = int((end_time - start_time) * 1000)

            # Calculate the difference between the two datasets
            difference = lerc_data - data

            # Calculate RMSE (Root Mean Square Error)
            rmse = np.sqrt(np.nanmean(difference**2))

        print(f"read time: {read_time_milliseconds} milliseconds")
        print(f"Root Mean Square Error (RMSE) between uncompressed and LERC: {rmse:.6f}")

        row = [
            precision_decimals,
            zstd_level,
            lerc_size_kb,
            compression_percent,
            write_time_milliseconds,
            read_time_milliseconds,
            rmse
        ]

        df = pd.concat([df, pd.DataFrame([row], columns=columns)])

df.to_csv(csv_filename, index=False)
