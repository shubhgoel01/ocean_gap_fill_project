import sys
import re

# distribution_fit.py
path1 = "c:/Users/SHUBH GOEL/Desktop/Projects/ocean-gap-fill-project-2/ocean_gapfill_mc/src/ocean_gapfill_mc/distribution_fit.py"
with open(path1, "r", encoding="utf-8") as f:
    content1 = f.read()

repl1 = """def build_cell_record(data_array: xr.DataArray, index_triplet: np.ndarray) -> dict:
    time_axis = data_array.get_axis_num("time")
    lat_axis = data_array.get_axis_num("lat")
    lon_axis = data_array.get_axis_num("lon")

    time_index = int(index_triplet[time_axis])
    lat_index = int(index_triplet[lat_axis])
    lon_index = int(index_triplet[lon_axis])"""
content1 = re.sub(r"def build_cell_record\(data_array: xr\.DataArray, index_triplet: np\.ndarray\) -> dict:\n    time_index = int\(index_triplet\[0\]\)\n    lat_index = int\(index_triplet\[1\]\)\n    lon_index = int\(index_triplet\[2\]\)", repl1, content1)
with open(path1, "w", encoding="utf-8") as f:
    f.write(content1)

# uncertainty.py
path2 = "c:/Users/SHUBH GOEL/Desktop/Projects/ocean-gap-fill-project-2/ocean_gapfill_mc/src/ocean_gapfill_mc/uncertainty.py"
with open(path2, "r", encoding="utf-8") as f:
    content2 = f.read()

repl2a = """    results = []
    
    time_axis = post_interpolation_data.get_axis_num("time")
    lat_axis = post_interpolation_data.get_axis_num("lat")
    lon_axis = post_interpolation_data.get_axis_num("lon")

    for cell in selected_cells:
        time_index, lat_index, lon_index = make_cell_key(cell)
        
        idx = [0] * post_interpolation_data.ndim
        idx[time_axis] = time_index
        idx[lat_axis] = lat_index
        idx[lon_axis] = lon_index
        idx = tuple(idx)
        
        results.append(
            {
                "time_index": time_index,
                "time_value": cell["time_value"],
                "lat_index": lat_index,
                "lat_value": cell["lat_value"],
                "lon_index": lon_index,
                "lon_value": cell["lon_value"],
                "mean": float(mean_map[idx]),
                "std": float(std_map[idx]),
                "lower_percentile": float(lower_percentile_map[idx]),
                "upper_percentile": float(upper_percentile_map[idx]),"""
content2 = re.sub(r"    results = \[\]\n    for cell in selected_cells:\n        time_index, lat_index, lon_index = make_cell_key\(cell\)\n        results\.append\(\n            {\n                \"time_index\": time_index,\n                \"time_value\": cell\[\"time_value\"\],\n                \"lat_index\": lat_index,\n                \"lat_value\": cell\[\"lat_value\"\],\n                \"lon_index\": lon_index,\n                \"lon_value\": cell\[\"lon_value\"\],\n                \"mean\": float\(mean_map\[time_index, lat_index, lon_index\]\),\n                \"std\": float\(std_map\[time_index, lat_index, lon_index\]\),\n                \"lower_percentile\": float\(lower_percentile_map\[time_index, lat_index, lon_index\]\),\n                \"upper_percentile\": float\(upper_percentile_map\[time_index, lat_index, lon_index\]\),", repl2a, content2)

repl2b = """def determine_cell_status(
    raw_observation_support: xr.DataArray,
    post_interpolation_data: xr.DataArray,
    time_index: int,
    lat_index: int,
    lon_index: int,
) -> str:
    \"\"\"Determine whether a cell had raw support, was interpolated, or needed Monte Carlo.

    The label refers to the explicit gap-filling stages on the final working
    grid, not to whether the target-grid cell was a direct raw observation.
    \"\"\"
    raw_support_value = raw_observation_support.isel(time=time_index, lat=lat_index, lon=lon_index).item()
    post_value = post_interpolation_data.isel(time=time_index, lat=lat_index, lon=lon_index).item()"""
content2 = re.sub(r"def determine_cell_status\(\n    raw_observation_support: xr\.DataArray,\n    post_interpolation_data: xr\.DataArray,\n    time_index: int,\n    lat_index: int,\n    lon_index: int,\n\) -> str:\n    \"\"\"Determine whether a cell had raw support, was interpolated, or needed Monte Carlo.\n\n    The label refers to the explicit gap-filling stages on the final working\n    grid, not to whether the target-grid cell was a direct raw observation.\n    \"\"\"\n    raw_support_value = raw_observation_support\.values\[time_index, lat_index, lon_index\]\n    post_value = post_interpolation_data\.values\[time_index, lat_index, lon_index\]", repl2b, content2)

with open(path2, "w", encoding="utf-8") as f:
    f.write(content2)

# monte_carlo.py
path3 = "c:/Users/SHUBH GOEL/Desktop/Projects/ocean-gap-fill-project-2/ocean_gapfill_mc/src/ocean_gapfill_mc/monte_carlo.py"
with open(path3, "r", encoding="utf-8") as f:
    content3 = f.read()

repl3_call = """        filled = write_samples_into_reconstructions(
            reconstructed_arrays,
            fit_result["cell"],
            sampled_values,
            data_array,
        )"""
content3 = re.sub(r"        filled = write_samples_into_reconstructions\(\n            reconstructed_arrays,\n            fit_result\[\"cell\"\],\n            sampled_values,\n        \)", repl3_call, content3)

repl3_func = """def write_samples_into_reconstructions(
    reconstructed_arrays: list[np.ndarray],
    cell: dict,
    sampled_values: np.ndarray,
    data_array: xr.DataArray,
) -> bool:
    \"\"\"Write one cell's simulated values into all reconstructed arrays.\"\"\"
    time_axis = data_array.get_axis_num("time")
    lat_axis = data_array.get_axis_num("lat")
    lon_axis = data_array.get_axis_num("lon")

    idx = [0] * data_array.ndim
    idx[time_axis] = int(cell["time_index"])
    idx[lat_axis] = int(cell["lat_index"])
    idx[lon_axis] = int(cell["lon_index"])
    idx = tuple(idx)

    wrote_any_value = False
    for simulation_index, array in enumerate(reconstructed_arrays):
        if simulation_index >= len(sampled_values):
            continue

        sampled_value = sampled_values[simulation_index]
        if not np.isfinite(sampled_value):
            continue

        array[idx] = float(sampled_value)
        wrote_any_value = True"""
content3 = re.sub(r"def write_samples_into_reconstructions\(\n    reconstructed_arrays: list\[np\.ndarray\],\n    cell: dict,\n    sampled_values: np\.ndarray,\n\) -> bool:\n    \"\"\"Write one cell's simulated values into all reconstructed arrays.\"\"\"\n    time_index = int\(cell\[\"time_index\"\]\)\n    lat_index = int\(cell\[\"lat_index\"\]\)\n    lon_index = int\(cell\[\"lon_index\"\]\)\n\n    wrote_any_value = False\n    for simulation_index, array in enumerate\(reconstructed_arrays\):\n        if simulation_index >= len\(sampled_values\):\n            continue\n\n        sampled_value = sampled_values\[simulation_index\]\n        if not np\.isfinite\(sampled_value\):\n            continue\n\n        array\[time_index, lat_index, lon_index\] = float\(sampled_value\)\n        wrote_any_value = True", repl3_func, content3)

with open(path3, "w", encoding="utf-8") as f:
    f.write(content3)

print("Replacement done")
