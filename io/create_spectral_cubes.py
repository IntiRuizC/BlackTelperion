import os
import glob
import numpy as np
import rasterio
from rasterio.transform import Affine
import spectral.io.envi as envi
import logging
import shutil

# Setup logger
logger = logging.getLogger(__name__)

def create_sentinel_spectral_cube(safe_directory, output_dir=None, bands_to_use=None, resolution=60):
    """
    Creates a spectral cube from Sentinel-2 L2A imagery in ENVI format (.dat and .hdr).
    
    This function processes Sentinel-2 L2A data to generate a spectral cube containing
    selected reflectance bands. The function locates the appropriate bands within the 
    SAFE directory structure, reads them, and combines them into a single multi-band image.
    The output is saved in ENVI format with a .dat data file and .hdr header file. 
    
    The function replaces the original nodata values (0) with NaN and converts 
    all data to float32 format for better processing compatibility.
    
    Parameters
    ----------
    safe_directory : str
        Path to the Sentinel-2 SAFE directory
        (e.g., "S2A_MSIL2A_20201210T185801_N0500_R113_T10TGK_20230302T053408.SAFE")
        Can be a full path to the .SAFE directory or to a subdirectory within it.
    
    output_dir : str, optional
        Directory where output ENVI files will be saved.
        If None, uses the same directory as the input SAFE file.
        The directory will be created if it doesn't exist.
    
    bands_to_use : list of str, optional
        List of reflectance band names to include in the spectral cube (e.g., ['B02', 'B03', 'B04']).
        If None or empty, all available reflectance bands for the specified resolution will be used.
        The order of bands in this list determines the order in the output spectral cube.
        
        Available bands by resolution:
        - 10m: B02, B03, B04, B08
        - 20m: B01, B02, B03, B04, B05, B06, B07, B8A, B11, B12
        - 60m: B01, B02, B03, B04, B05, B06, B07, B8A, B09, B11, B12
    
    resolution : int, optional
        Resolution in meters. Must be one of: 10, 20, or 60.
        Default is 60m, which includes the most available bands.
        
    Returns
    -------
    tuple
        (spectral_cube, metadata, output_path)
        - spectral_cube: numpy.ndarray
            3D array with dimensions (height, width, bands)
        - metadata: dict
            ENVI header metadata
        - output_path: str
            Path to the output files (without extension)
        
    Raises
    ------
    ValueError
        If bands are not available at the specified resolution or if resolution is invalid.
        The error message will list which bands are available at the given resolution.
    
    FileNotFoundError
        If required files or directories cannot be found within the SAFE structure.
        
    Examples
    --------
    >>> # Create RGB cube at 60m resolution
    >>> cube, metadata, path = create_sentinel_spectral_cube(
    ...     "S2A_MSIL2A_20201210T185801_N0500_R113_T10TGK_20230302T053408.SAFE",
    ...     bands_to_use=['B04', 'B03', 'B02'],
    ...     resolution=60
    ... )
    
    >>> # Use all available bands at 10m resolution
    >>> cube, metadata, path = create_sentinel_spectral_cube(
    ...     "S2A_MSIL2A_20201210T185801_N0500_R113_T10TGK_20230302T053408.SAFE",
    ...     resolution=10
    ... )
    
    Notes
    -----
    - The output file naming follows the pattern: {base_name}_spectralcube_R{resolution}m.dat
      where base_name is derived from the SAFE directory name.
    - The spectral cube shape is (height, width, bands) with bands in the last dimension.
    - All data is converted to float32 and 0 values are replaced with NaN.
    - The ENVI file is saved in BSQ (Band Sequential) format.
    """
    # Validate resolution
    if resolution not in [10, 20, 60]:
        logger.error(f"Invalid resolution: {resolution}. Must be 10, 20, or 60.")
        raise ValueError(f"Invalid resolution: {resolution}. Must be 10, 20, or 60.")
    
    # Set default output directory if not provided
    if output_dir is None:
        # Use the same directory as the input SAFE file
        output_dir = os.path.dirname(safe_directory)
        logger.info(f"No output directory specified. Using: {output_dir}")
    
    # Make sure the output directory exists
    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
            logger.info(f"Created output directory: {output_dir}")
        except Exception as e:
            logger.error(f"Failed to create output directory: {str(e)}")
            raise
    
    # Get the base name from the SAFE directory path
    # First, check if the path itself ends with .SAFE
    if safe_directory.endswith('.SAFE'):
        # Extract just the filename part without the .SAFE extension
        base_name = os.path.basename(safe_directory[:-5])
    else:
        # The path might be to a subdirectory within the SAFE structure
        # Find the .SAFE part in the path
        safe_parts = safe_directory.split('.SAFE')
        if len(safe_parts) > 1:  # If .SAFE exists in the path
            # Get just the filename part of the path leading up to .SAFE
            safe_dir_part = safe_parts[0]
            base_name = os.path.basename(safe_dir_part)
        else:
            # Fallback: just use the basename of the provided path
            base_name = os.path.basename(safe_directory)
    
    logger.info(f"Using base name: {base_name} for output files")
    
    # Create output filename with the specified format - note the "*" around "spectralcube"
    output_filename = f"{base_name}_spectralcube_R{resolution}m"
    full_output_path = os.path.join(output_dir, output_filename)
    
    logger.info(f"Output will be saved as: {full_output_path}.dat and {full_output_path}.hdr")
    logger.info(f"Creating spectral cube with reflectance bands: {bands_to_use} at {resolution}m resolution")
    
    # Dictionary of available reflectance bands for each resolution
    available_bands = {
        10: ['B02', 'B03', 'B04', 'B08'],
        
        20: ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B8A', 'B11', 'B12'],
        
        60: ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B8A', 'B09', 'B11', 'B12']
    }
    
    # If bands_to_use is None or empty, use all available bands for the resolution
    if not bands_to_use:
        bands_to_use = available_bands[resolution]
        logger.info(f"No specific bands requested. Using all available bands at {resolution}m: {bands_to_use}")
    
    # Check if all requested bands are available in the specified resolution
    invalid_bands = []
    for band in bands_to_use:
        if band not in available_bands[resolution]:
            invalid_bands.append(f"{band} (not available at {resolution}m resolution)")
    
    if invalid_bands:
        error_msg = f"The following bands are not available at {resolution}m resolution: {', '.join(invalid_bands)}. "
        error_msg += f"Available bands for {resolution}m resolution are: {', '.join(available_bands[resolution])}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    if invalid_bands:
        error_msg = f"The following bands are not available at {resolution}m resolution: {', '.join(invalid_bands)}. "
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    # Find the correct image data directory
    granule_dir = glob.glob(os.path.join(safe_directory, 'GRANULE', 'L2A_*'))
    if not granule_dir:
        error_msg = f"Could not find GRANULE directory in {safe_directory}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
    
    # Construct the path to the resolution-specific image directory
    img_data_dir = os.path.join(granule_dir[0], 'IMG_DATA', f'R{resolution}m')
    if not os.path.exists(img_data_dir):
        error_msg = f"Could not find R{resolution}m directory in {granule_dir[0]}/IMG_DATA"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
    
    logger.info(f"Found image data directory: {img_data_dir}")
    
    # Dictionary to store paths for each band
    band_paths = {}
    
    # Find band files in the directory
    for band in bands_to_use:
        # Search pattern for Sentinel-2 L2A band files
        pattern = os.path.join(img_data_dir, f"*_{band}_{resolution}m.jp2")
        matches = glob.glob(pattern)
        
        if matches:
            band_paths[band] = matches[0]
            logger.info(f"Found {band} at: {band_paths[band]}")
        else:
            error_msg = f"Could not find band {band} in {img_data_dir}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
    
    # Load the first band to get reference metadata
    reference_band = bands_to_use[0]
    with rasterio.open(band_paths[reference_band]) as src:
        reference_profile = src.profile.copy()
        reference_shape = (src.height, src.width)
        
        # Extract geospatial information
        transform = src.transform
        crs = src.crs
    
    logger.info(f"Reference band {reference_band} shape: {reference_shape}")
    
    # Create an empty array for our spectral cube
    num_bands = len(bands_to_use)
    spectral_cube = np.zeros((reference_shape[0], reference_shape[1], num_bands), dtype=np.float32)
    
    # Load each band and add to the spectral cube
    for i, band in enumerate(bands_to_use):
        with rasterio.open(band_paths[band]) as src:
            band_data = src.read(1).astype(np.float32)
            
            # Replace nodata (0) with NaN
            band_data[band_data == 0] = np.nan
            
            # Add to spectral cube - bands in the last dimension
            spectral_cube[:, :, i] = band_data
    
    logger.info(f"Spectral cube created with shape: {spectral_cube.shape}")
    
    # Prepare ENVI header metadata
    envi_metadata = {
        'description': f'Sentinel-2 L2A spectral cube with bands: {", ".join(bands_to_use)}',
        'bands': num_bands,
        'lines': reference_shape[0],
        'samples': reference_shape[1],
        'interleave': 'bsq',  # Explicitly set BSQ interleave format
        'data type': 4,       # 4 corresponds to float32
        'byte order': 0,
        'band names': bands_to_use,
        'coordinate system string': str(crs),
        'map info': [
            str(crs.to_epsg()),
            '1', '1',  # Pixel location (ENVI format is 1-based)
            str(transform.c),  # Upper left x
            str(transform.f),  # Upper left y
            str(transform.a),  # Pixel width
            str(abs(transform.e)),  # Pixel height
            ' '
        ],
        'data ignore value': float('nan')  # Set NaN as the nodata value
    }
    
    # Save as ENVI format
    try:
        # Create full file paths with correct extensions
        base_output_path = full_output_path
        header_path = f"{base_output_path}.hdr"
        dat_path = f"{base_output_path}.dat"
        
        # Save the spectral cube using spectral.io.envi with explicit extension
        envi.save_image(header_path, spectral_cube, metadata=envi_metadata, 
                       force=True, ext='.dat')
        
        # Check if the file was saved properly
        if os.path.exists(dat_path):
            logger.info(f"Saved spectral cube to {dat_path} with header {header_path}")
        else:
            logger.warning(f"Could not find {dat_path}, checking for alternate extensions...")
            # Find the created data file (which might have .img extension)
            img_path = None
            potential_extensions = ['.img', '.IMG', '.raw', '.RAW', '.bil', '.BIL', '.bsq', '.BSQ', '.bip', '.BIP']
            for ext in potential_extensions:
                temp_path = f"{base_output_path}{ext}"
                if os.path.exists(temp_path):
                    img_path = temp_path
                    break
            
            # If found, rename to .dat
            if img_path:
                shutil.move(img_path, dat_path)
                logger.info(f"Renamed data file from {img_path} to {dat_path}")
            else:
                logger.warning(f"Could not find data file with known extensions. Please check the output directory.")
        
    except Exception as e:
        logger.error(f"Failed to save ENVI files: {str(e)}")
        raise
    
    return spectral_cube, envi_metadata, full_output_path
        
        
import rasterio
from rasterio.enums import Resampling
import re

def create_hyspex_spectral_cube(hdr_path, output_dir):
    """
    Processes a HySpex hyperspectral cube (.hdr and .bsq files):
    1. Reads the .hdr file and corresponding .bsq data
    2. Converts all values <= 0 OR equal to 15000 to np.nan
    3. Updates the header to set NaN as the data ignore value
    4. Saves the result as .dat and .hdr files
    
    Parameters:
    -----------
    hdr_path : str
        Path to the .hdr file
    output_dir : str
        Directory to save output files
    
    Returns:
    --------
    tuple
        Paths to output .dat and .hdr files
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract base name and directory
    base_name = os.path.basename(hdr_path).replace('.hdr', '')
    input_dir = os.path.dirname(hdr_path)
    
    # Find the corresponding .bsq file
    bsq_path = os.path.join(input_dir, f"{base_name}.bsq")
    if not os.path.exists(bsq_path):
        # Try finding any .bsq file in the same directory with the same base name
        for file in os.listdir(input_dir):
            if file.startswith(base_name) and file.endswith('.bsq'):
                bsq_path = os.path.join(input_dir, file)
                break
    
    if not os.path.exists(bsq_path):
        error_msg = f"Could not find .bsq file for {hdr_path}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
    
    logger.info(f"Found corresponding BSQ file: {bsq_path}")
    
    # Output paths
    output_dat_path = os.path.join(output_dir, f"{base_name}.dat")
    output_hdr_path = os.path.join(output_dir, f"{base_name}.hdr")
    
    # Read the original header file content
    with open(hdr_path, 'r') as f:
        orig_hdr_content = f.read()
    
    # Open dataset with rasterio
    try:
        with rasterio.open(bsq_path) as src:
            # Get metadata
            profile = src.profile
            
            # Update profile for output
            profile.update(
                driver='ENVI',
                dtype=np.float32,
                nodata=np.nan
            )
            
            logger.info(f"Processing hyperspectral cube with dimensions: {src.width}x{src.height}x{src.count}")
            
            # Create output dataset
            with rasterio.open(output_dat_path, 'w', **profile) as dst:
                # Process each band
                for band_idx in range(1, src.count + 1):
                    # Read band data
                    data = src.read(band_idx)
                    
                    # Convert to float32
                    data = data.astype(np.float32)
                    
                    # Set values <= 0 OR equal to 15000 to NaN
                    data[(data <= 0) | (data == 15000)] = np.nan
                    
                    # Write to output
                    dst.write(data, band_idx)
    except Exception as e:
        error_msg = f"Error processing BSQ file: {str(e)}"
        logger.error(error_msg)
        raise
    
    logger.info(f"Data processing complete. Now updating header file...")
    
    # Check if header file exists
    if os.path.exists(output_hdr_path):
        # Read the current header
        with open(output_hdr_path, 'r') as f:
            new_hdr_content = f.read()
    else:
        logger.warning(f"Header file not created by rasterio. Creating from the original header...")
        # If not, copy from the original
        shutil.copy(hdr_path, output_hdr_path)
        with open(output_hdr_path, 'r') as f:
            new_hdr_content = f.read()
    
    # Define mandatory fields
    mandatory_fields = [
        'bands', 'header offset', 'data type', 'interleave', 'byte order',
        'map info', 'coordinate system string', 'pixel size', 'background',
        'wavelength units', 'band names', 'wavelength', 'fwhm'
    ]
    
    # Update data ignore value
    new_hdr_content = re.sub(
        r'data ignore value\s*=\s*[^\n]*',
        'data ignore value = nan',
        new_hdr_content
    )
    
    # Ensure data type is 4 (float32)
    new_hdr_content = re.sub(
        r'data type\s*=\s*\d+',
        'data type = 4',
        new_hdr_content
    )
    
    # Check for missing mandatory fields
    for field in mandatory_fields:
        # Skip data ignore value as we've already set it
        if field == 'data ignore value':
            continue
            
        # Check if the field exists in the new header
        if not re.search(rf'{field}\s*=', new_hdr_content, re.IGNORECASE):
            # If not, extract it from the original header
            pattern = rf'{field}\s*=\s*[^{{}}]*(?:{{[^{{}}]*}})?'
            match = re.search(pattern, orig_hdr_content, re.IGNORECASE | re.DOTALL)
            if match:
                # Add the field to the new header
                field_content = match.group(0)
                new_hdr_content += f"\n{field_content}"
                logger.info(f"Added missing field: {field}")
            else:
                logger.warning(f"Could not find required field '{field}' in original header")
    
    # Write the updated header
    try:
        with open(output_hdr_path, 'w') as f:
            f.write(new_hdr_content)
    except Exception as e:
        error_msg = f"Failed to write header file: {str(e)}"
        logger.error(error_msg)
        raise
    
    logger.info(f"Processing complete!")
    logger.info(f"Output data file: {output_dat_path}")
    logger.info(f"Output header file: {output_hdr_path}")
    
    return output_dat_path, output_hdr_path

