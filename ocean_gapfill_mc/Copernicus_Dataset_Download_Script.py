import copernicusmarine
import os
import calendar

# --- Configuration ---
# The exact gap-free dataset ID from your URL
dataset_id = "cmems_obs-oc_glo_bgc-plankton_my_l4-gapfree-multi-4km_P1D"
year = 2020
months = list(range(12, 13))
output_directory = f"./data/target_data/2022"

# Your specific geographical bounding box
min_lon = 40.0
max_lon = 120.0
min_lat = -30.0
max_lat = 30.0

os.makedirs(output_directory, exist_ok=True)
print(f"Starting Level-4 Gap-Free subset downloads for {year}, months {months}...")

for month in months:
    _, last_day = calendar.monthrange(year, month)
    print(f"\nStarting {year}-{month:02d}...")

    for day in range(1, last_day + 1):
        start_date = f"{year}-{month:02d}-{day:02d}T00:00:00"
        end_date = f"{year}-{month:02d}-{day:02d}T23:59:59"
        filename = f"{year}_{month:02d}_{day:02d}.nc"

        print(f"Fetching {filename}...")

        try:
            # Requesting the geographical subset directly from the Marine server
            copernicusmarine.subset(
                dataset_id=dataset_id,
                variables=["CHL"], # Ensure this matches the variable name in the L4 dataset (usually CHL)
                minimum_longitude=min_lon,
                maximum_longitude=max_lon,
                minimum_latitude=min_lat,
                maximum_latitude=max_lat,
                start_datetime=start_date,
                end_datetime=end_date,
                output_filename=filename,
                output_directory=output_directory
            )
        except Exception as e:
            print(f" -> Error downloading {filename}: {e}")

    print(f"{month:02d} {year} L-4 extraction complete!")

print(f"\nAll requested {year} L-4 extractions complete!")
