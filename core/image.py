import numpy as np
from typing import Dict, Any, Optional, Union, Tuple, List, Sequence
import os
import logging

class Image:
    """
    Base class for representing and manipulating remote sensing imagery data.
    
    The Image class provides a foundation for working with multi-band remote sensing data
    by encapsulating both the pixel values and validity masks. It offers methods for 
    band extraction, spatial subsetting, mask application, and general array operations.
    
    This class focuses purely on array operations and basic data handling,
    completely independent of platform-specific details or metadata.
    
    Attributes:
        data (numpy.ndarray): 3D array containing the image data with dimensions (height, width, bands).
            Accessible via the .data property.
        mask (numpy.ndarray): 2D boolean array indicating valid pixels (True=valid, False=invalid).
            Accessible via the .mask property.
        shape (tuple): Tuple of (height, width, bands) dimensions.
        height (int): Number of rows in the image.
        width (int): Number of columns in the image.
        num_bands (int): Number of spectral bands in the image.
        
    Examples:
        >>> # Load an image from ENVI format
        >>> img = Image.from_envi("path/to/image.hdr")
        >>> band1 = img.get_band(0)  # Get first band
    """
    
    def __init__(
        self, 
        data: np.ndarray, 
        mask: Optional[np.ndarray] = None
    ):
        """
        Initialize a new Image object.
        
        Args:
            data: NumPy array containing the image data with shape (height, width, bands)
                or (height, width) for single-band images. Will be converted to float32
                if not already a floating-point type.
            mask: Optional binary mask indicating valid pixels (True=valid, False=invalid).
                If not provided, a mask will be created where all non-NaN values are
                considered valid.
        
        Raises:
            TypeError: If data is not a NumPy array.
            ValueError: If data dimensions are not 2D or 3D.
            ValueError: If provided mask shape does not match data dimensions.
            
        Examples:
            >>> import numpy as np
            >>> # Create a simple image with all valid pixels
            >>> data = np.ones((10, 10, 3), dtype=np.float32)
            >>> img = Image(data)
            >>> 
            >>> # Create an image with a custom mask
            >>> data = np.random.random((20, 20, 4))  # 4-band image
            >>> mask = np.ones((20, 20), dtype=bool)  # All pixels valid
            >>> mask[0:5, 0:5] = False  # Mark top-left corner as invalid
            >>> img = Image(data, mask)
        """
        # Ensure data is a numpy array
        if not isinstance(data, np.ndarray):
            raise TypeError("Data must be a numpy array")
        
        # Convert to float32 if not already a floating-point type
        if not np.issubdtype(data.dtype, np.floating):
            self._data = data.astype(np.float32)
        else:
            self._data = data
        
        # Handle single-band data (ensuring 3D array)
        if data.ndim == 2:
            # Reshape to (height, width, 1) for single-band data
            self._data = self._data.reshape(*self._data.shape, 1)
        elif data.ndim != 3:
            raise ValueError(f"Data must be 2D or 3D, got {data.ndim}D")
        
        # Set mask (all valid if not provided)
        if mask is None:
            # Create mask where non-NaN values are valid
            self._mask = ~np.isnan(self._data).any(axis=2)
        else:
            if mask.shape != (self._data.shape[0], self._data.shape[1]):
                raise ValueError(f"Mask shape {mask.shape} must match data dimensions ({self._data.shape[0]}, {self._data.shape[1]})")
            self._mask = mask.astype(bool)
        
        # Set up logging
        self._logger = logging.getLogger(__name__)
    
    @classmethod
    def from_envi(cls, file_path: str) -> 'Image':
        """
        Create an Image object from ENVI format files (.hdr and data file).
        
        This method reads both the header file (.hdr) and the corresponding binary data file
        (.dat, .bsq, .bil, .bip, or extension-less) to create an Image object.
        
        Args:
            file_path: Path to either the .hdr file or the data file. The method will
                    find the corresponding pair file automatically.
                    
        Returns:
            A new Image object with data loaded from the ENVI files.
            
        Raises:
            FileNotFoundError: If either the header file or data file cannot be found.
            ValueError: If the header file is not a valid ENVI header.
            
        Examples:
            >>> # Load from header file
            >>> img = Image.from_envi("/path/to/landsat_image.hdr")
            >>> 
            >>> # Load from data file (will find corresponding .hdr file)
            >>> img = Image.from_envi("/path/to/sentinel2_image.dat")
        """
        # Get base path without extension
        base_path = os.path.splitext(file_path)[0]
        hdr_path = f"{base_path}.hdr"
        
        # Check if header file exists
        if not os.path.exists(hdr_path):
            raise FileNotFoundError(f"Header file not found: {hdr_path}")
        
        # Try common data file extensions
        data_extensions = ['.dat', '.bsq', '.bil', '.bip']
        data_path = None
        
        for ext in data_extensions:
            potential_path = f"{base_path}{ext}"
            if os.path.exists(potential_path):
                data_path = potential_path
                break
        
        # If no data file found, use extension-less file if it exists
        if data_path is None and os.path.exists(base_path) and os.path.isfile(base_path):
            data_path = base_path
        
        # If still no data file found, raise error
        if data_path is None:
            raise FileNotFoundError(f"Data file not found for header: {hdr_path}")
        
        # Parse header file to get dimensions and data type
        dimensions, data_type, interleave, header_offset, byte_order, data_ignore_value = cls._parse_envi_header_basic(hdr_path)
        
        # Read binary data file
        data = cls._read_envi_data_basic(
            data_path, 
            dimensions, 
            data_type, 
            interleave, 
            header_offset, 
            byte_order,
            data_ignore_value
        )
        
        # Create image object (with no metadata)
        return cls(data)
    
    @staticmethod
    def _parse_envi_header_basic(hdr_path: str) -> Tuple[Dict[str, int], int, str, int, int, Optional[float]]:
        """
        Parse an ENVI header file to extract basic information needed for data reading.
        
        This internal method extracts essential parameters from the ENVI header file
        that are required to properly read the binary data file.
        
        Args:
            hdr_path: Path to the ENVI header file (.hdr)
                
        Returns:
            Tuple containing:
                - dimensions dict with keys 'width', 'height', 'bands'
                - data type code (ENVI data type)
                - interleave format ('bsq', 'bil', or 'bip')
                - header offset in bytes
                - byte order (0=little endian, 1=big endian)
                - data ignore value (value to be replaced with NaN, or None if not specified)
                
        Raises:
            ValueError: If the file is not a valid ENVI header (must start with 'ENVI').
        """
        # Default values
        dimensions = {'width': 0, 'height': 0, 'bands': 1}
        data_type = 4  # Default to float32
        interleave = "bsq"
        header_offset = 0
        byte_order = 0
        data_ignore_value = np.nan
        
        with open(hdr_path, 'r') as f:
            lines = f.readlines()
        
        # Check if this is an ENVI header
        if not lines[0].strip().startswith('ENVI'):
            raise ValueError(f"File {hdr_path} is not a valid ENVI header")
        
        # Parse key-value pairs
        for line in lines[1:]:  # Skip ENVI line
            line = line.strip()
            
            if line == "" or '=' not in line:
                continue
                
            key, value = line.split('=', 1)
            key = key.strip().lower()
            value = value.strip()
            
            # Extract essential information
            if key == 'samples':
                dimensions['width'] = int(value)
            elif key == 'lines':
                dimensions['height'] = int(value)
            elif key == 'bands':
                dimensions['bands'] = int(value)
            elif key == 'data type':
                data_type = int(value)
            elif key == 'interleave':
                interleave = value.lower().strip()
            elif key == 'header offset':
                header_offset = int(value)
            elif key == 'byte order':
                byte_order = int(value)
            elif key == 'data ignore value':
                try:
                    data_ignore_value = float(value)
                except ValueError:
                    pass
        
        return dimensions, data_type, interleave, header_offset, byte_order, data_ignore_value
    
    @staticmethod
    def _read_envi_data_basic(
        dat_path: str, 
        dimensions: Dict[str, int], 
        data_type: int, 
        interleave: str, 
        header_offset: int, 
        byte_order: int,
        data_ignore_value: Optional[float]
    ) -> np.ndarray:
        """
        Read binary data from an ENVI data file.
        
        This internal method reads the binary data from an ENVI data file based on the
        parameters extracted from the header file.
        
        Args:
            dat_path: Path to the ENVI data file (.dat, .bsq, .bil, .bip, or extension-less)
            dimensions: Dictionary with keys 'width', 'height', 'bands' specifying image dimensions
            data_type: ENVI data type code (1=byte, 2=int16, 4=float32, etc.)
            interleave: Interleave format ('bsq', 'bil', 'bip')
            header_offset: Offset in bytes from the start of the file
            byte_order: Byte order (0=little endian, 1=big endian)
            data_ignore_value: Value to be replaced with NaN, or None if not specified
                
        Returns:
            NumPy array containing the image data with shape (height, width, bands)
            
        Raises:
            ValueError: If the interleave format is not recognized.
            IOError: If there is an error reading the binary data file.
        """
        width = dimensions['width']
        height = dimensions['height']
        bands = dimensions['bands']
        
        # Map ENVI data types to numpy data types
        envi_to_numpy_dtype = {
            1: np.uint8,
            2: np.int16,
            3: np.int32,
            4: np.float32,
            5: np.float64,
            6: np.complex64,
            9: np.complex128,
            12: np.uint16,
            13: np.uint32,
            14: np.int64,
            15: np.uint64
        }
        
        # Get the numpy dtype
        dtype = envi_to_numpy_dtype.get(data_type, np.float32)
        
        # Set endianness
        endian = '<' if byte_order == 0 else '>'
        
        # Simpler approach using dtype character codes
        dtype_char = np.dtype(dtype).char  # Get the type character code
        full_dtype = np.dtype(endian + dtype_char)  # Add endianness to the char code
        
        # Read binary data
        with open(dat_path, 'rb') as f:
            # Apply header offset if needed
            if header_offset > 0:
                f.seek(header_offset)
            
            # Read raw data
            raw_data = np.fromfile(f, dtype=full_dtype)
        
        # Reshape data to (height, width, bands)
        interleave = interleave.lower()
        
        if interleave == "bsq":  # Band Sequential (band, height, width)
            data = raw_data.reshape((bands, height, width))
            data = np.transpose(data, (1, 2, 0))  # Convert to (height, width, bands)
        elif interleave == "bil":  # Band Interleaved by Line (height, band, width)
            data = raw_data.reshape((height, bands, width))
            data = np.transpose(data, (0, 2, 1))  # Convert to (height, width, bands)
        elif interleave == "bip":  # Band Interleaved by Pixel (height, width, band)
            data = raw_data.reshape((height, width, bands))
        else:
            raise ValueError(f"Unknown interleave format: {interleave}")
        
        # Convert to float32 and handle no data values
        if data.dtype != np.float32:
            data = data.astype(np.float32)
        
        if data_ignore_value is not None:
            data[data == data_ignore_value] = np.nan
        
        return data
    
    @property
    def data(self) -> np.ndarray:
        """
        Get the image data array.
        
        Returns:
            3D NumPy array of shape (height, width, bands) containing the image data.
            For single-band images, the shape is still 3D: (height, width, 1).
        """
        return self._data
    
    @property
    def mask(self) -> np.ndarray:
        """
        Get the valid data mask.
        
        Returns:
            2D boolean NumPy array of shape (height, width) where True indicates
            valid pixels and False indicates invalid/masked pixels.
        """
        return self._mask
    
    @property
    def shape(self) -> Tuple[int, int, int]:
        """
        Get the shape of the image data.
        
        Returns:
            Tuple of (height, width, bands) representing the dimensions of the image.
        """
        return self._data.shape
    
    @property
    def num_bands(self) -> int:
        """
        Get the number of spectral bands in the image.
        
        Returns:
            Integer representing the number of bands.
        """
        return self._data.shape[2]
    
    @property
    def height(self) -> int:
        """
        Get the height of the image in pixels.
        
        Returns:
            Integer representing the number of rows in the image.
        """
        return self._data.shape[0]
    
    @property
    def width(self) -> int:
        """
        Get the width of the image in pixels.
        
        Returns:
            Integer representing the number of columns in the image.
        """
        return self._data.shape[1]
    
    def get_band(self, band_index: int) -> np.ndarray:
        """
        Extract a single band from the image.
        
        Args:
            band_index: Zero-based index of the band to retrieve.
                
        Returns:
            2D NumPy array of shape (height, width) containing the requested band data.
            
        Raises:
            IndexError: If band_index is out of range (0 to num_bands-1).
            
        Examples:
            >>> img = Image.from_envi("multispectral_image.hdr")
            >>> # Get the first band (index 0)
            >>> nir_band = img.get_band(0)
            >>> # Compute statistics on the band
            >>> mean_value = np.nanmean(nir_band)
        """
        if band_index < 0 or band_index >= self.num_bands:
            raise IndexError(f"Band index {band_index} out of range (0-{self.num_bands-1})")
        
        return self._data[:, :, band_index]
    
    def get_bands(self, band_indices: Sequence[int]) -> np.ndarray:
        """
        Extract multiple bands from the image.
        
        Args:
            band_indices: Sequence (list, tuple, array) of zero-based indices
                        of the bands to retrieve.
                
        Returns:
            3D NumPy array of shape (height, width, len(band_indices)) containing
            the requested band data.
            
        Raises:
            IndexError: If any band index is out of range (0 to num_bands-1).
            
        Examples:
            >>> img = Image.from_envi("landsat_image.hdr")
            >>> # Get RGB bands (assuming they are at indices 2, 1, 0)
            >>> rgb_bands = img.get_bands([2, 1, 0])
        """
        # Validate indices
        for idx in band_indices:
            if idx < 0 or idx >= self.num_bands:
                raise IndexError(f"Band index {idx} out of range (0-{self.num_bands-1})")
        
        return self._data[:, :, band_indices]
    
    
    def apply_mask(self, mask: np.ndarray) -> 'Image':
        """
        Apply a new mask to the image in-place.
        
        This method updates the image's internal mask that identifies valid pixels.
        A mask value of True indicates a valid pixel; False indicates an invalid pixel.
        
        Args:
            mask: Boolean array indicating valid pixels (shape must match image dimensions).
                
        Returns:
            Self, to allow method chaining.
            
        Raises:
            ValueError: If mask shape does not match image dimensions (height, width).
            
        Examples:
            >>> img = Image.from_envi("satellite_image.hdr")
            >>> # Create a custom mask (e.g., cloud mask)
            >>> cloud_mask = np.ones((img.height, img.width), dtype=bool)
            >>> cloud_mask[cloud_areas] = False
            >>> # Apply mask in-place
            >>> img.apply_mask(cloud_mask)
        """
        if mask.shape != (self.height, self.width):
            raise ValueError(f"Mask shape {mask.shape} must match image dimensions ({self.height}, {self.width})")
        
        # Update existing mask instead of creating new Image
        self._mask = mask.astype(bool)
        return self
    
    def apply_function(self, func, *args, inplace: bool = False, **kwargs) -> 'Image':
        """
        Apply a function to the image data.
        
        This method applies a function to the entire image data array. The function should
        accept a NumPy array as its first argument and return a NumPy array.
        
        Args:
            func: Function to apply to the data. Should accept a NumPy array as its first
                argument and return a transformed NumPy array.
            *args: Additional positional arguments to pass to the function.
            inplace: If True (default), modifies the image in-place and returns self.
                  If False, returns a new Image object with the function applied.
            **kwargs: Additional keyword arguments to pass to the function.
                
        Returns:
            Image object (self if inplace=True, or a new object if inplace=False).
            
        Examples:
            >>> img = Image.from_envi("spectral_image.hdr")
            >>> # Apply log transformation in-place
            >>> img.apply_function(np.log1p)
            >>> 
            >>> # Apply normalization and create a new image
            >>> def normalize(array):
            >>>     return (array - np.nanmin(array)) / (np.nanmax(array) - np.nanmin(array))
            >>> normalized_img = img.apply_function(normalize, inplace=False)
        """
        result_data = func(self._data, *args, **kwargs)
        
        # Ensure result is in the correct shape
        if result_data.ndim == 2:
            result_data = result_data.reshape(*result_data.shape, 1)
        
        if inplace:
            # Update existing data in-place
            self._data = result_data
            return self
        else:
            # Return a new Image with the result data
            return Image(result_data, self._mask.copy())
    
    
    def copy(self) -> 'Image':
        """
        Create a deep copy of the image.
        
        This method creates a new Image object with copies of the data and mask arrays,
        ensuring that modifications to the new object do not affect the original.
        
        Returns:
            New Image object with identical but independent data and mask.
            
        Examples:
            >>> original = Image.from_envi("image.hdr")
            >>> duplicate = original.copy()
            >>> # Modify duplicate without affecting original
            >>> duplicate.apply_function(lambda x: x * 2, inplace=True)
        """
        return Image(self._data.copy(), self._mask.copy())
    
    def __repr__(self) -> str:
        """
        Return a string representation of the Image object.
        
        This method defines how the object is displayed when printed or in interactive sessions.
        
        Returns:
            String describing the Image with its shape and data type.
        """
        return f"Image(shape={self.shape}, dtype={self._data.dtype})"
        
